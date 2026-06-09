import numpy as np
import random
import torch
from sps.spike_utils import TransformationRule
from sps.snp_system import SNPSystem  
from sps.config import Config


class MSNPSystemDivModGPU:
    def __init__(self, configurationVector, spikingVector, spikingTransitionMatrix, 
                 ruleVector, synapsesMatrix, max_steps=1000, deterministic=True, 
                 single_spike_train=None, input_neurons=None, output_neurons=None, targetVector=None, 
                 applyingRuleVector=None, device='cpu',testsize=1):

        self.max_steps = max_steps #TODO check this 3 attributes
        self.output_neurons = output_neurons
        self.input_neurons = input_neurons

        print("input neurons", self.input_neurons, "output neurons", self.output_neurons)

        # Set device and dtype based on device
        if device == 'gpu' or device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.dtype = torch.float32
                print(f"MSNPSystem using device: CUDA GPU with dtype={self.dtype}")
            else:
                print("Warning: CUDA not available. Falling back to CPU.")
                self.device = torch.device('cpu')
                self.dtype = torch.int32
                print(f"MSNPSystem using device: CPU with dtype={self.dtype}")
        else:  # device == 'cpu'
            self.device = torch.device('cpu')
            self.dtype = torch.int32
            print(f"MSNPSystem using device: CPU with dtype={self.dtype}")

        if applyingRuleVector is None or configurationVector is None or \
            spikingTransitionMatrix is None or ruleVector is None:
            raise ValueError("ApplyingRuleVector, configurationVector, spikingTransitionMatrix and ruleVector cannot be None")
        
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        rule_num = len(spikingTransitionMatrix)
        neuron_num = len(configurationVector)
        
        self.testsize = testsize
        self.pooling_image = torch.zeros((Config.NEURONS_L3, testsize), dtype=self.dtype, device='cpu') if output_neurons is not None else None
        self.deterministic = deterministic
        
        # Convert to PyTorch tensors with appropriate dtype
        self.configurationVector = torch.tensor(configurationVector, dtype=self.dtype, device=self.device)
        self.spikingTransitionMatrix = torch.tensor(spikingTransitionMatrix, dtype=self.dtype, device=self.device)
        self.synapsesMatrix = torch.tensor(synapsesMatrix, dtype=self.dtype, device=self.device)
        self.netGainVector = torch.zeros(neuron_num, dtype=self.dtype, device=self.device)
        self.ruleVector = torch.tensor(ruleVector, dtype=self.dtype, device=self.device)
        self.applyingRuleVector = torch.tensor(applyingRuleVector, dtype=torch.int32, device=self.device)
        
        self.sMpi = self.spikingTransitionMatrix * self.synapsesMatrix
        
        if spikingVector is None:
            self.spikingVector = torch.zeros(rule_num, dtype=self.dtype, device=self.device)
        else:
            self.spikingVector = torch.tensor(spikingVector, dtype=self.dtype, device=self.device)
        
        if single_spike_train is not None:
            self.single_spike_train = torch.tensor(single_spike_train, dtype=self.dtype, device=self.device)
        else:
            self.single_spike_train = torch.tensor([], dtype=self.dtype, device=self.device)
        
        # Target Vector: monodimensional vector of length n
        # where each entry is the number of spikes produced by the rule if it's a firing rule
        if targetVector is not None:
            self.targetVector = torch.tensor(targetVector, dtype=self.dtype, device=self.device)
        else:
            self.targetVector = torch.zeros(rule_num, dtype=self.dtype, device=self.device)
        
        # Initialize the time step counter
        self.t_step = 0
    
    # Input images as spike trains, in the format (num_images, num_input_neurons)
    # For example, for 28x28 images, num_input_neurons would be 784 
    # and each image would be represented as a vector of length 784
    def loadImages(self, img_spike_train):
        """Load images as spike trains for CNN mode"""
        # Reshape da (N, 28, 28) a (N, 784) if necessary
        if len(img_spike_train.shape) == 3:
            img_spike_train = img_spike_train.reshape(img_spike_train.shape[0], -1)
        self.img_spike_train = torch.tensor(img_spike_train, dtype=self.dtype, device=self.device)
    
    def step(self, verbose=False):
        """Execute one step of the system"""
        
        # SPIKE TRAIN INPUT
        # CNN -> Images
        if Config.MODE == "CNN":
            if self.t_step < self.img_spike_train.shape[0]:
                self.configurationVector[self.input_neurons] += self.img_spike_train[self.t_step] #TODO lasciato questa linea, it should work :X
                #self.configurationVector[self.input_neurons, :784] += self.img_spike_train[self.t_step]
                if verbose:
                    print(f"Applied image spike train at step {self.t_step + 1}: added {self.img_spike_train[self.t_step]} spikes to input neurons {self.input_neurons.cpu().numpy()}")
        
        # SINGLE SPIKE TRAIN -> boolean spike train for all input neurons
        elif self.single_spike_train.size(0) > 0 and self.t_step < self.single_spike_train.shape[0]:
            if self.dtype == torch.int32:
                spike_value = 1
            else:
                spike_value = 1.0
                
            if self.single_spike_train[self.t_step] == spike_value:
                self.configurationVector[self.input_neurons] += spike_value
                if verbose:
                    print(f"Applied spike train at step {self.t_step + 1}: added {spike_value} spike to input neurons {self.input_neurons.cpu().numpy()}")
        
        # UPDATE SPIKING VECTOR
        self.update_spiking_vector(verbose=verbose)
        
        # CRUCIAL STEP -> UPDATE CONFIGURATION VECTOR AND NET GAIN VECTOR
        self.netGainVector = self.spikingVector @ self.spikingTransitionMatrix
        self.configurationVector = self.configurationVector + self.netGainVector
        
        # White Hole not implemented
        # if Config.WHITE_HOLE:
        #     self.configurationVector.zero_()
        #     if verbose:
        #         print("White hole applied: configuration vector reset to zero")
        if self.pooling_image is not None and Config.NUM_LAYERS - 3 < self.t_step <= self.testsize + Config.NUM_LAYERS - 3:
            #self.pooling_image[self.t_step] = self.configurationVector[self.output_neurons] TODO check this, fix hardcoded values
            self.pooling_image[:, self.t_step - Config.NUM_LAYERS + 2] = self.configurationVector[6192:7544] #see "self.pooling_image" in snp_system.py
        self.t_step += 1
        print("time step", self.t_step) #TODo delete
        return True
    
    def execute(self, verbose=False, startAgain=True):
        """Execute the system until halt condition is met"""
        if startAgain:
            self.t_step = 0
        
        if verbose:
            print("Initial Configuration Vector:", self.configurationVector.cpu().numpy())
            print("-" * 30)
        
        # determine input length based on mode and available spike trains
        if Config.MODE == "CNN":
            input_length = self.img_spike_train.shape[0]
        else:
            input_length = self.single_spike_train.shape[0] if self.single_spike_train.size(0) > 0 else 0
        
        while self.step(verbose=verbose) and (self.t_step < self.max_steps or self.t_step < input_length):
            if verbose:
                print("Step:", self.t_step + 1)
                print("Spiking Vector applied:", self.spikingVector.cpu().numpy())
                print("Configuration Vector obtained:", self.configurationVector.cpu().numpy())
                print("Net Gain Vector in step", self.t_step + 1, ":", self.netGainVector.cpu().numpy())
                print("-" * 30)
            
            # Check halt condition
            if torch.all(self.spikingVector == 0) and (self.t_step >= input_length):
                print("Computation halts because the spiking vector is zero; no more rules can be applied; the input is accepted")
                return True
                    
        print("Computation halts because the maximum number of steps has been reached; the input is rejected")
        return False
    
    """
    - charge: the current charge of the neuron
    - source: the number of spikes consumed by the rule
    - div: the divisor in the rule's regular expression (a^div)*
    - mod: the modulus in the rule's regular expression (a^mod)
    - target: the number of spikes produced by the rule (if > 0) or 0 for forgetting rules
    """
    def rule_check(self, charge, source, div, mod, target):
        # Convert to Python scalars if they are tensors
        if isinstance(charge, torch.Tensor):
            charge = charge.item()
        if isinstance(source, torch.Tensor):
            source = source.item()
        if isinstance(div, torch.Tensor):
            div = div.item()
        if isinstance(mod, torch.Tensor):
            mod = mod.item()
        if isinstance(target, torch.Tensor):
            target = target.item()
        
        # Adjust for dtype (int vs float)
        if self.dtype == torch.int32:
            # Integer comparisons
            if charge > 0 and charge >= mod and charge >= target:
                if div > 0:
                    return charge >= source and (charge - mod) % div == 0
                elif div == 0:
                    return charge >= source and charge == mod
        else:
            # Float comparisons (with small epsilon for numerical stability)
            eps = 1e-6
            if charge > 0 and charge >= mod - eps and charge >= target - eps:
                if div > 0:
                    # For floats, modulo operation is different
                    # Check if (charge - mod) is divisible by div
                    remainder = (charge - mod) % div
                    return charge >= source - eps and (remainder < eps or abs(remainder - div) < eps)
                elif div == 0:
                    return charge >= source - eps and abs(charge - mod) < eps
        return False
    
    def update_spiking_vector(self, verbose=False):
        """Update spiking vector based on current configuration and rules"""

        rule_num = len(self.spikingTransitionMatrix)
        
        # Reset spiking vector
        self.spikingVector.zero_()
        
        # Apply rules - versione vettoriale
        for i in range(rule_num):
            # Get the source spikes (absolute value of spikingTransitionMatrix)
            source = torch.abs(self.spikingTransitionMatrix[i][self.applyingRuleVector[i]])
            
            # Get rule parameters
            rule_div = self.ruleVector[i][0]
            rule_mod = self.ruleVector[i][1]
            
            neuron_idx = self.applyingRuleVector[i]  
            charge = self.configurationVector[neuron_idx]  
            
            # Get target (spikes produced)
            target = self.targetVector[i]
            
            # Check if rule applies
            if self.rule_check(charge, source, rule_div, rule_mod, target):
                if self.dtype == torch.int32:
                    self.spikingVector[i] = 1
                else:
                    self.spikingVector[i] = 1.0
            
    
    def get_configuration_vector(self):
        """Return configuration vector as NumPy array (copy from device)"""
        return self.configurationVector.cpu().numpy()
    
    def get_spiking_vector(self):
        """Return spiking vector as NumPy array (copy from device)"""
        return self.spikingVector.cpu().numpy()
    
    def get_net_gain_vector(self):
        """Return net gain vector as NumPy array (copy from device)"""
        return self.netGainVector.cpu().numpy()
    
    def get_spiking_transition_matrix(self):
        """Return spiking transition matrix as NumPy array"""
        return self.spikingTransitionMatrix.cpu().numpy()
    
    def get_rule_vector(self):
        """Return rule vector as NumPy array"""
        return self.ruleVector.cpu().numpy()
    
    def get_applying_rule_vector(self):
        """Return applying rule vector as NumPy array"""
        return self.applyingRuleVector.cpu().numpy()
    
    def get_target_vector(self):
        """Return target vector as NumPy array"""
        return self.targetVector.cpu().numpy()
    
    def to(self, device):
        """Move the entire system to a specific device (CPU or GPU) with appropriate dtype"""
        new_device = torch.device(device)
        
        # Determine new dtype based on device
        if new_device.type == 'cuda':
            new_dtype = torch.float32
        else:
            new_dtype = torch.int32
        
        # If dtype changes, we need to recreate tensors
        if new_dtype != self.dtype:
            print(f"Changing dtype from {self.dtype} to {new_dtype} for device {new_device}")
            self.dtype = new_dtype
            
            self.configurationVector = self.configurationVector.to(dtype=self.dtype, device=new_device)
            self.spikingTransitionMatrix = self.spikingTransitionMatrix.to(dtype=self.dtype, device=new_device)
            self.synapsesMatrix = self.synapsesMatrix.to(dtype=self.dtype, device=new_device)
            self.netGainVector = self.netGainVector.to(dtype=self.dtype, device=new_device)
            self.ruleVector = self.ruleVector.to(dtype=self.dtype, device=new_device)
            self.sMpi = self.sMpi.to(dtype=self.dtype, device=new_device)
            self.spikingVector = self.spikingVector.to(dtype=self.dtype, device=new_device)
            self.single_spike_train = self.single_spike_train.to(dtype=self.dtype, device=new_device)
            self.targetVector = self.targetVector.to(dtype=self.dtype, device=new_device)
            if hasattr(self, 'img_spike_train'):
                self.img_spike_train = self.img_spike_train.to(dtype=self.dtype, device=new_device)
        else:
            # Just move to new device
            self.configurationVector = self.configurationVector.to(new_device)
            self.spikingTransitionMatrix = self.spikingTransitionMatrix.to(new_device)
            self.synapsesMatrix = self.synapsesMatrix.to(new_device)
            self.netGainVector = self.netGainVector.to(new_device)
            self.ruleVector = self.ruleVector.to(new_device)
            self.sMpi = self.sMpi.to(new_device)
            self.spikingVector = self.spikingVector.to(new_device)
            self.single_spike_train = self.single_spike_train.to(new_device)
            self.targetVector = self.targetVector.to(new_device)
            if hasattr(self, 'img_spike_train'):
                self.img_spike_train = self.img_spike_train.to(new_device)
        
        # These always stay as int32
        self.applyingRuleVector = self.applyingRuleVector.to(new_device)
        
        self.device = new_device
        return self
    
    def __str__(self):
        """String representation of the system state"""
        return (f"Device: {self.device} (dtype: {self.dtype})\n"
                f"Deterministic: {self.deterministic}\n"
                f"SpikingTransitionMatrix:\n{self.get_spiking_transition_matrix()}\n"
                f"Input Neurons: {self.input_neurons}\n"
                f"Output Neurons: {self.output_neurons}\n"
                f"Configuration Vector: {self.get_configuration_vector()}\n"
                f"Spiking Vector: {self.get_spiking_vector()}\n"
                f"Net Gain Vector: {self.get_net_gain_vector()}\n"
                f"Rule Vector: {self.get_rule_vector()}\n"
                f"Target Vector: {self.get_target_vector()}\n"
                f"Applying Rule Vector: {self.get_applying_rule_vector()}")