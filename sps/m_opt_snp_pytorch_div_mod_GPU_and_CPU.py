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
                 applyingRuleVector=None, device='cpu', testsize=1):

        self.max_steps = max_steps
        self.output_neurons = output_neurons
        self.input_neurons = input_neurons

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
        self.pooling_image = torch.zeros((len(output_neurons), self.testsize), dtype=self.dtype, device='cpu') if output_neurons is not None else None
        self.deterministic = deterministic
        
        # Enable cuDNN benchmark mode for optimal GPU performance
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # Convert to PyTorch tensors with appropriate dtype
        self.configurationVector = torch.tensor(configurationVector, dtype=self.dtype, device=self.device)
        self.spikingTransitionMatrix = torch.tensor(spikingTransitionMatrix, dtype=self.dtype, device=self.device)
        self.synapsesMatrix = torch.tensor(synapsesMatrix, dtype=self.dtype, device=self.device)
        self.netGainVector = torch.zeros(neuron_num, dtype=self.dtype, device=self.device)
        self.ruleVector = torch.tensor(ruleVector, dtype=self.dtype, device=self.device)
        self.applyingRuleVector = torch.tensor(applyingRuleVector, dtype=torch.int32, device=self.device)
        
        # Pre-compute synapses * spiking transition matrix
        self.sMpi = self.spikingTransitionMatrix * self.synapsesMatrix
        
        if spikingVector is None:
            self.spikingVector = torch.zeros(rule_num, dtype=self.dtype, device=self.device)
        else:
            self.spikingVector = torch.tensor(spikingVector, dtype=self.dtype, device=self.device)
        
        if single_spike_train is not None:
            self.single_spike_train = torch.tensor(single_spike_train, dtype=self.dtype, device=self.device)
        else:
            self.single_spike_train = torch.tensor([], dtype=self.dtype, device=self.device)
        
        # Target Vector
        if targetVector is not None:
            self.targetVector = torch.tensor(targetVector, dtype=self.dtype, device=self.device)
        else:
            self.targetVector = torch.zeros(rule_num, dtype=self.dtype, device=self.device)
        
        # Pre-allocate tensors for vectorized operations
        self.rule_indices = torch.arange(rule_num, device=self.device)
        self._temp_charges = torch.zeros(rule_num, dtype=self.dtype, device=self.device)
        self._temp_sources = torch.zeros(rule_num, dtype=self.dtype, device=self.device)
        
        # Extract rule parameters once (they don't change)
        self.rule_divs = self.ruleVector[:, 0]  # (rule_num,)
        self.rule_mods = self.ruleVector[:, 1]  # (rule_num,)
        
        # Initialize the time step counter
        self.t_step = 0
    
    def rule_check_vectorized(self, charges, sources, divs, mods, targets):
        """Fully vectorized rule check for GPU acceleration"""
        eps = 1e-6 if self.dtype != torch.int32 else 0
        
        # Basic conditions
        charge_positive = charges > 0
        charge_ge_mod = charges >= mods - eps
        charge_ge_target = charges >= targets - eps
        charge_ge_source = charges >= sources - eps
        
        # Case div > 0
        div_positive = divs > 0
        # Handle division by zero safely
        safe_divs = torch.where(div_positive, divs, torch.ones_like(divs))
        remainder = torch.where(div_positive, (charges - mods) % safe_divs, torch.zeros_like(charges))
        
        if self.dtype == torch.int32:
            div_condition = (remainder == 0)
        else:
            div_condition = (remainder < eps) | (torch.abs(remainder - safe_divs) < eps)
        
        # Case div == 0
        div_zero = ~div_positive
        if self.dtype == torch.int32:
            mod_condition = (charges == mods)
        else:
            mod_condition = torch.abs(charges - mods) < eps
        
        # Combine all conditions
        result = charge_positive & charge_ge_mod & charge_ge_target & (
            (div_positive & div_condition & charge_ge_source) |
            (div_zero & mod_condition & charge_ge_source)
        )
        
        return result
    
    # Input images as spike trains
    def loadImages(self, img_spike_train):
        """Load images as spike trains for CNN mode"""
        # Reshape from (N, 28, 28) to (N, 784) if necessary
        if len(img_spike_train.shape) == 3:
            img_spike_train = img_spike_train.reshape(img_spike_train.shape[0], -1)
        self.img_spike_train = torch.tensor(img_spike_train, dtype=self.dtype, device=self.device)
    
    def step(self, verbose=False):
        """Execute one step of the system - GPU optimized"""
        
        # SPIKE TRAIN INPUT
        # CNN -> Images
        if Config.MODE == "CNN":
            if self.t_step < self.img_spike_train.shape[0]:
                self.configurationVector[self.input_neurons] += self.img_spike_train[self.t_step]
                if verbose:
                    print(f"Applied image spike train at step {self.t_step + 1}: added spikes to input neurons")
        
        # SINGLE SPIKE TRAIN -> boolean spike train for all input neurons
        elif self.single_spike_train.size(0) > 0 and self.t_step < self.single_spike_train.shape[0]:
            spike_value = 1 if self.dtype == torch.int32 else 1.0
                
            if self.single_spike_train[self.t_step] == spike_value:
                self.configurationVector[self.input_neurons] += spike_value
                if verbose:
                    print(f"Applied spike train at step {self.t_step + 1}: added {spike_value} spike to input neurons")
        
        # UPDATE SPIKING VECTOR - Fully vectorized
        self.update_spiking_vector(verbose=verbose)
        
        # CRUCIAL STEP -> UPDATE CONFIGURATION VECTOR AND NET GAIN VECTOR
        self.netGainVector = self.spikingVector @ self.spikingTransitionMatrix
        self.configurationVector = self.configurationVector + self.netGainVector
        
        # Handle pooling for CNN output
        if self.pooling_image is not None and Config.NUM_LAYERS - 3 < self.t_step <= self.testsize + Config.NUM_LAYERS - 3:
            # Transfer to CPU only if necessary (pooling_image is on CPU)
            if self.pooling_image.device != self.configurationVector.device:
                self.pooling_image[:, self.t_step - Config.NUM_LAYERS + 2] = \
                    self.configurationVector[self.output_neurons].cpu()
            else:
                self.pooling_image[:, self.t_step - Config.NUM_LAYERS + 2] = \
                    self.configurationVector[self.output_neurons]
        
        self.t_step += 1
        return True
    
    def execute(self, verbose=False, startAgain=True):
        """Execute the system until halt condition is met"""
        if startAgain:
            self.t_step = 0
        
        if verbose:
            print("Initial Configuration Vector:", self.configurationVector.cpu().numpy())
            print("-" * 30)
        
        # Determine input length based on mode and available spike trains
        if Config.MODE == "CNN":
            input_length = self.img_spike_train.shape[0] if hasattr(self, 'img_spike_train') else 0
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
    
    def update_spiking_vector(self, verbose=False):
        """Update spiking vector based on current configuration and rules - Fully vectorized for GPU"""
        
        # Get charges for all applicable neurons at once
        neuron_indices = self.applyingRuleVector  # (rule_num,)
        charges = self.configurationVector[neuron_indices]  # (rule_num,)
        
        # Get sources for all rules at once
        # source = abs(spikingTransitionMatrix[i][applyingRuleVector[i]])
        sources = torch.abs(self.spikingTransitionMatrix[self.rule_indices, neuron_indices])
        
        # Apply vectorized rule check
        mask = self.rule_check_vectorized(
            charges, 
            sources, 
            self.rule_divs, 
            self.rule_mods, 
            self.targetVector
        )
        
        # Update spiking vector based on mask
        if self.dtype == torch.int32:
            self.spikingVector = mask.int()
        else:
            self.spikingVector = mask.float()
        
        if verbose:
            active_rules = torch.where(mask)[0]
            if len(active_rules) > 0:
                print(f"Active rules: {active_rules.cpu().numpy()}")
    
    # Utility methods for accessing system state
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
            
            # Move all tensors with new dtype
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
            
            # Re-pre-allocate temp tensors with new dtype
            self._temp_charges = self._temp_charges.to(dtype=self.dtype, device=new_device)
            self._temp_sources = self._temp_sources.to(dtype=self.dtype, device=new_device)
            
            # Update pre-extracted parameters
            self.rule_divs = self.ruleVector[:, 0]
            self.rule_mods = self.ruleVector[:, 1]
        else:
            # Just move to new device, keep same dtype
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
            
            # Move temp tensors
            self._temp_charges = self._temp_charges.to(new_device)
            self._temp_sources = self._temp_sources.to(new_device)
            
            # Update pre-extracted parameters
            self.rule_divs = self.ruleVector[:, 0]
            self.rule_mods = self.ruleVector[:, 1]
        
        # These always stay as int32
        self.applyingRuleVector = self.applyingRuleVector.to(new_device)
        self.rule_indices = self.rule_indices.to(new_device)
        
        self.device = new_device
        
        # Re-enable cuDNN benchmark if moving to GPU
        if new_device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
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