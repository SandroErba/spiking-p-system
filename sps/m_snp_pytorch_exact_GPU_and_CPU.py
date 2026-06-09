import numpy as np
import random
import torch
from sps.spike_utils import TransformationRule
from sps.snp_system import SNPSystem  
from sps.config import Config


class MSNPSystemExactGPU:
    
    def __init__(self, configurationVector, spikingVector, spikingTransitionMatrix, 
                 synapsesMatrix, ruleVector, max_steps=1000, deterministic=True, 
                 single_spike_train=None, input_neurons=None, output_neurons=None,
                 applyingRuleVector=None, device='cpu',testsize=1):
        
        # Set device (GPU if available, CPU otherwise) and dtype based on device
        # For CPU, int32 is often more efficient for this type of computation; 
        # for GPU, float32 allows parallelization
        if device == 'gpu' or device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.dtype = torch.float32
                print(f"MSNPSystemGPU using device: CUDA GPU with dtype={self.dtype}")
            else:
                print("Warning: CUDA not available. Falling back to CPU.")
                self.device = torch.device('cpu')
                self.dtype = torch.int32
                print(f"MSNPSystemGPU using device: CPU with dtype={self.dtype}")
        else:  # device == 'cpu'
            self.device = torch.device('cpu')
            self.dtype = torch.int32
            print(f"MSNPSystemGPU using device: CPU with dtype={self.dtype}")

        if applyingRuleVector is None or configurationVector is None or \
            spikingTransitionMatrix is None or ruleVector is None:
            raise ValueError("ApplyingRuleVector, configurationVector, spikingTransitionMatrix and ruleVector cannot be None")
        
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        
        rule_num = len(spikingTransitionMatrix)
        neuron_num = len(configurationVector)
        
        self.testsize = testsize
        self.pooling_image = torch.zeros((len(output_neurons), self.testsize), dtype=self.dtype, device='cpu') if output_neurons is not None else None
        
        # Numpy arrays
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.max_steps = max_steps
        self.deterministic = deterministic
        
        self.configurationVector = torch.tensor(configurationVector, dtype=self.dtype, device=self.device)
        self.spikingTransitionMatrix = torch.tensor(spikingTransitionMatrix, dtype=self.dtype, device=self.device)
        self.synapsesMatrix = torch.tensor(synapsesMatrix, dtype=self.dtype, device=self.device)
        self.netGainVector = torch.zeros(neuron_num, dtype=self.dtype, device=self.device)
        self.ruleVector = torch.tensor(ruleVector, dtype=self.dtype, device=self.device)
        self.applyingRuleVector = torch.tensor(applyingRuleVector, dtype=torch.int32, device=self.device)
        
        # sMpi inherits the same dtype and device
        self.sMpi = self.spikingTransitionMatrix * self.synapsesMatrix
        
        # bincount requires int32
        self.ruleCountPerNeuron = torch.bincount(self.applyingRuleVector, minlength=neuron_num)
        
        if spikingVector is None:
            self.spikingVector = torch.zeros(rule_num, dtype=self.dtype, device=self.device)
        else:
            self.spikingVector = torch.tensor(spikingVector, dtype=self.dtype, device=self.device)
        
        if single_spike_train is not None:
            self.single_spike_train = torch.tensor(single_spike_train, dtype=self.dtype, device=self.device)
        else:
            self.single_spike_train = torch.tensor([], dtype=self.dtype, device=self.device)
        
        # Initialize the time step counter
        self.t_step = 0
    
    def loadImages(self, img_spike_train):
        """Load images as spike trains for CNN mode"""
        if len(img_spike_train.shape) == 3:
            img_spike_train = img_spike_train.reshape(img_spike_train.shape[0], -1)
        self.img_spike_train = torch.tensor(img_spike_train, dtype=self.dtype, device=self.device)

    def step(self, verbose=False):
        """Execute one step of the system"""
        
        # SPIKE TRAIN INPUT
        if Config.MODE == "CNN":
            if self.t_step < self.img_spike_train.shape[0]:
                self.configurationVector[self.input_neurons] += self.img_spike_train[self.t_step]
                if verbose:
                    print(f"Applied image spike train at step {self.t_step + 1}")
        
        elif self.single_spike_train.size(0) > 0 and self.t_step < self.single_spike_train.shape[0]:
            if self.dtype == torch.int32:
                spike_value = 1
            else:
                spike_value = 1.0
                
            if self.single_spike_train[self.t_step] == spike_value:
                self.configurationVector[self.input_neurons] += spike_value
                if verbose:
                    print(f"Applied spike train at step {self.t_step + 1}")
        
        extendedConfigVector = torch.zeros_like(self.spikingVector, dtype=self.dtype, device=self.device)
        idx = 0
        for i in range(len(self.configurationVector)):
            count = self.ruleCountPerNeuron[i].item()
            extendedConfigVector[idx:idx+count] = self.configurationVector[i]
            idx += count
        
        diff = torch.abs(extendedConfigVector - self.ruleVector)
        
        self.spikingVector = torch.div(1, 1 + diff, rounding_mode='floor') if self.dtype == torch.int32 else torch.floor(1.0 / (1.0 + diff))
        
        self.netGainVector = self.spikingVector @ self.sMpi
        self.configurationVector = self.configurationVector + self.netGainVector
        
        # White hole not used in this implementation
        # if Config.WHITE_HOLE:
        #     self.configurationVector = torch.zeros_like(self.configurationVector, device=self.device)
        #     if verbose:
        #         print("White hole applied: configuration vector reset to zero")
        
        
        if verbose:
            print(self)

        if self.pooling_image is not None:
            self.pooling_image[self.t_step] = self.configurationVector[self.output_neurons]
        self.t_step += 1
        return True
    
    def execute(self, verbose=False, startAgain=True):
        """Execute the system until halt condition is met"""
        if startAgain:
            self.t_step = 0
        
        if verbose:
            print("Initial Configuration Vector:", self.configurationVector.cpu().numpy())
            print("-" * 30)
        
        # determine input length based on mode
        if Config.MODE == "CNN":
            input_length = self.img_spike_train.shape[0]
        else:
            input_length = len(self.single_spike_train) if hasattr(self.single_spike_train, '__len__') else 0
        
        while self.step(verbose=verbose) and (self.t_step < self.max_steps or self.t_step < input_length):
            if verbose:
                print("Step:", self.t_step + 1)
                print("Spiking Vector applied:", self.spikingVector.cpu().numpy())
                print("Configuration Vector obtained:", self.configurationVector.cpu().numpy())
                print("Net Gain Vector in step", self.t_step + 1, ":", self.netGainVector.cpu().numpy())
                print("-" * 30)
            
            # Check halt condition (spikingVector == 0 in modo appropriato al dtype)
            if torch.all(self.spikingVector == 0) and (self.t_step >= input_length):
                print("Computation halts: spiking vector is zero, input is accepted")
                return True
            
        
        print("Computation halts: maximum number of steps reached, input is rejected")
        return False
    
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
        return self.sMpi.cpu().numpy()

    def get_rule_vector(self):
        return self.ruleVector.cpu().numpy()

    def get_applying_rule_vector(self):
        return self.applyingRuleVector.cpu().numpy()

    def to(self, device):
        """Sposta l'intero sistema su un device specifico (CPU o GPU) con dtype appropriato"""
        new_device = torch.device(device)
        
        # Determina il nuovo dtype in base al device
        if new_device.type == 'cuda':
            new_dtype = torch.float32
        else:
            new_dtype = torch.int32
        
        # Se il dtype cambia, dobbiamo ricreare i tensori
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
            if hasattr(self, 'img_spike_train'):
                self.img_spike_train = self.img_spike_train.to(dtype=self.dtype, device=new_device)
        else:
            self.configurationVector = self.configurationVector.to(new_device)
            self.spikingTransitionMatrix = self.spikingTransitionMatrix.to(new_device)
            self.synapsesMatrix = self.synapsesMatrix.to(new_device)
            self.netGainVector = self.netGainVector.to(new_device)
            self.ruleVector = self.ruleVector.to(new_device)
            self.sMpi = self.sMpi.to(new_device)
            self.spikingVector = self.spikingVector.to(new_device)
            self.single_spike_train = self.single_spike_train.to(new_device)
            if hasattr(self, 'img_spike_train'):
                self.img_spike_train = self.img_spike_train.to(new_device)
        
        self.applyingRuleVector = self.applyingRuleVector.to(new_device)        
        self.device = new_device
        return self
    
    def __str__(self):
        """String representation of the system state"""
        return (f"Device: {self.device} (dtype: {self.dtype})\n"
                f"Deterministic: {self.deterministic}\n"
                f"Synapses Spiking Transition Matrix:\n{self.get_spiking_transition_matrix()}\n"
                f"Input Neurons: {self.input_neurons}\n"
                f"Output Neurons: {self.output_neurons}\n"
                f"Configuration Vector: {self.get_configuration_vector()}\n"
                f"Spiking Vector: {self.get_spiking_vector()}\n"
                f"Net Gain Vector: {self.get_net_gain_vector()}\n"
                f"Rule Vector: {self.get_rule_vector()}\n"
                f"Applying Rule Vector: {self.get_applying_rule_vector()}\n")

