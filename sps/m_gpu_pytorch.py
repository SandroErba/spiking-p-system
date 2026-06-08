import numpy as np
import random
import torch
from sps.spike_utils import TransformationRule
from sps.snp_system import SNPSystem  
from sps.config import Config


class MSNPSystemGPU:
    
    def __init__(self, configurationVector, spikingVector, spikingTransitionMatrix, 
                 synapsesMatrix, ruleVector, max_steps=1000, deterministic=True, 
                 single_spike_train=None, input_neurons=None, 
                 applyingRuleVector=None, device='cpu'):
        
        # Configura il device (GPU se disponibile, altrimenti CPU)
        if device == 'gpu' or device == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"MSNPSystemGPU using device: CUDA GPU")
            else:
                print("Warning: CUDA not available. Falling back to CPU.")
                self.device = torch.device('cpu')
        else:  # device == 'cpu'
            self.device = torch.device('cpu')
            print(f"MSNPSystemGPU using device: CPU")

        if applyingRuleVector is None or configurationVector is None or \
            spikingTransitionMatrix is None or ruleVector is None:
            raise ValueError("ApplyingRuleVector, configurationVector, spikingTransitionMatrix and ruleVector cannot be None")
        
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        
        rule_num = len(spikingTransitionMatrix)
        neuron_num = len(configurationVector)
        
        self.max_steps = max_steps
        self.deterministic = deterministic
        
        # Convert to PyTorch tensors (automaticamente sul device corretto)
        self.configurationVector = torch.tensor(configurationVector, dtype=torch.int32, device=self.device)
        self.spikingTransitionMatrix = torch.tensor(spikingTransitionMatrix, dtype=torch.int32, device=self.device)
        self.synapsesMatrix = torch.tensor(synapsesMatrix, dtype=torch.int32, device=self.device)
        self.netGainVector = torch.zeros(neuron_num, dtype=torch.int32, device=self.device)
        self.ruleVector = torch.tensor(ruleVector, dtype=torch.int32, device=self.device)
        self.applyingRuleVector = torch.tensor(applyingRuleVector, dtype=torch.int32, device=self.device)
        
        # ⚠️ FIX: Converti sMpi a float32 UNA VOLTA SOLA all'inizio (evita bottleneck)
        self.sMpi = (self.spikingTransitionMatrix * self.synapsesMatrix).to(torch.float32)
        
        self.ruleCountPerNeuron = torch.bincount(self.applyingRuleVector, minlength=neuron_num)
        
        if spikingVector is None:
            self.spikingVector = torch.zeros(rule_num, dtype=torch.int32, device=self.device)
        else:
            self.spikingVector = torch.tensor(spikingVector, dtype=torch.int32, device=self.device)
        
        if input_neurons is None:
            self.input_neurons = torch.tensor([], dtype=torch.int32, device=self.device)
        else:
            self.input_neurons = torch.tensor(input_neurons, dtype=torch.int32, device=self.device)
        
        if single_spike_train is not None:
            self.single_spike_train = torch.tensor(single_spike_train, dtype=torch.int32, device=self.device)
        else:
            self.single_spike_train = torch.tensor([], dtype=torch.int32, device=self.device)
        
        # Initialize the time step counter
        self.t_step = 0
    
    def loadImages(self, img_spike_train):
        """Load images as spike trains for CNN mode"""
        if len(img_spike_train.shape) == 3:
            img_spike_train = img_spike_train.reshape(img_spike_train.shape[0], -1)
        self.img_spike_train = torch.tensor(img_spike_train, dtype=torch.int32, device=self.device)
    
    def step(self, verbose=False):
        """Execute one step of the system"""
        
        # SPIKE TRAIN INPUT
        if Config.MODE == "CNN":
            if self.t_step < self.img_spike_train.shape[0]:
                self.configurationVector[self.input_neurons] += self.img_spike_train[self.t_step]
                if verbose:
                    print(f"Applied image spike train at step {self.t_step + 1}")
        
        elif self.single_spike_train.size(0) > 0 and self.t_step < self.single_spike_train.shape[0]:
            if self.single_spike_train[self.t_step] == 1:
                self.configurationVector[self.input_neurons] += 1
                if verbose:
                    print(f"Applied spike train at step {self.t_step + 1}")
        
        # Extended configuration vector
        extendedConfigVector = torch.zeros_like(self.spikingVector, dtype=torch.int32, device=self.device)
        idx = 0
        for i in range(len(self.configurationVector)):
            count = self.ruleCountPerNeuron[i].item()
            extendedConfigVector[idx:idx+count] = self.configurationVector[i]
            idx += count
        
        # Rule application
        diff = torch.abs(extendedConfigVector - self.ruleVector)
        denominator = 1 + diff
        self.spikingVector = torch.div(1, denominator, rounding_mode='floor')
        
        # ⚠️ FIX: conversione a float solo per la moltiplicazione (sMpi è già float32)
        # La conversione to(float32) è veloce perché crea una view, non copia
        net_gain_float = self.spikingVector.to(torch.float32) @ self.sMpi
        self.netGainVector = net_gain_float.to(torch.int32)
        self.configurationVector = self.configurationVector + self.netGainVector
        
        if Config.WHITE_HOLE:
            self.configurationVector = torch.zeros_like(self.configurationVector, device=self.device)
            if verbose:
                print("White hole applied: configuration vector reset to zero")
        
        if verbose:
            print(self)

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
            
            # Check halt condition
            if torch.all(self.spikingVector == 0) and (self.t_step >= input_length):
                print("Computation halts: spiking vector is zero, input is accepted")
                return True
            
            self.t_step += 1
        
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
    
    def to(self, device):
        """Sposta l'intero sistema su un device specifico (CPU o GPU)"""
        self.device = torch.device(device)
        self.configurationVector = self.configurationVector.to(self.device)
        self.spikingTransitionMatrix = self.spikingTransitionMatrix.to(self.device)
        self.synapsesMatrix = self.synapsesMatrix.to(self.device)
        self.netGainVector = self.netGainVector.to(self.device)
        self.ruleVector = self.ruleVector.to(self.device)
        self.applyingRuleVector = self.applyingRuleVector.to(self.device)
        self.sMpi = self.sMpi.to(self.device)
        self.spikingVector = self.spikingVector.to(self.device)
        self.input_neurons = self.input_neurons.to(self.device)
        self.single_spike_train = self.single_spike_train.to(self.device)
        if hasattr(self, 'img_spike_train'):
            self.img_spike_train = self.img_spike_train.to(self.device)
        return self
    
    def __str__(self):
        """String representation of the system state"""
        config_np = self.configurationVector.cpu().numpy()
        spiking_np = self.spikingVector.cpu().numpy()
        netgain_np = self.netGainVector.cpu().numpy()
        sMPi = self.sMpi.cpu().numpy()
        applyingRuleVector_np = self.applyingRuleVector.cpu().numpy()
        
        return (f"Device: {self.device}\n"
                f"Deterministic: {self.deterministic}\n"
                f"Synapses Spiking Transition Matrix:\n{sMPi}\n"
                f"Input Neurons: {self.input_neurons.cpu().numpy()}\n"
                f"Configuration Vector: {config_np}\n"
                f"Spiking Vector: {spiking_np}\n"
                f"Net Gain Vector: {netgain_np}\n"
                f"Rule Vector: {self.ruleVector.cpu().numpy()}\n"
                f"Applying Rule Vector: {applyingRuleVector_np}\n")