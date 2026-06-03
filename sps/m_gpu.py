import numpy as np
import random
import cupy as cp
from sps.spike_utils import TransformationRule
from sps.snp_system import SNPSystem  
from sps.config import Config


class MSNPSystemGPU:
       
    def __init__(self, configurationVector, spikingVector, spikingTransitionMatrix, synapsesMatrix, ruleVector, max_steps=1000, deterministic=True, 
                 single_spike_train=None, input_neurons=None, 
                 applyingRuleVector=None):
        
        if applyingRuleVector is None or configurationVector is None or spikingTransitionMatrix is None or ruleVector is None:
            raise ValueError("configurationVector, spikingTransitionMatrix and ruleVector cannot be None")
        
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        
        

        rule_num = len(spikingTransitionMatrix)
        neuron_num = len(configurationVector)

        self.max_steps = max_steps
        self.deterministic = deterministic

        # Convert to CuPy arrays (automatically on GPU)
        self.configurationVector = cp.asarray(configurationVector, dtype=cp.int32)
        self.spikingTransitionMatrix = cp.asarray(spikingTransitionMatrix, dtype=cp.int32)
        self.synapsesMatrix = cp.asarray(synapsesMatrix, dtype=cp.int32)
        self.netGainVector = cp.zeros(neuron_num, dtype=cp.int32)
        self.ruleVector = cp.asarray(ruleVector, dtype=cp.int32)  # rule vector is in format (r1, r2, r3, r4, rn)
        # each entry is the index of the neuron to which the rule applies
        self.applyingRuleVector = cp.asarray(applyingRuleVector, dtype=cp.int32)
        self.sMpi = spikingTransitionMatrix * synapsesMatrix
        self.ruleCountPerNeuron = cp.bincount(self.applyingRuleVector, minlength=neuron_num)

        if spikingVector is None:
            self.spikingVector = cp.zeros(rule_num, dtype=cp.int32)
        else:
            self.spikingVector = cp.asarray(spikingVector, dtype=cp.int32)


        if input_neurons is None:
            self.input_neurons = cp.asarray([], dtype=cp.int32)
        else:
            self.input_neurons = cp.asarray(input_neurons, dtype=cp.int32)

        if single_spike_train is not None:
            self.single_spike_train = cp.asarray(single_spike_train, dtype=cp.int32)
        else:
            self.single_spike_train = cp.asarray([], dtype=cp.int32)
        
        # Initialize the time step counter
        self.t_step = 0

    def loadImages(self, img_spike_train):
        """Load images as spike trains for CNN mode"""
        if len(img_spike_train.shape) == 3:
            img_spike_train = img_spike_train.reshape(img_spike_train.shape[0], -1)
        self.img_spike_train = cp.asarray(img_spike_train, dtype=cp.int32)

    def step(self, verbose=False):
        """Execute one step of the system"""
        
        # SPIKE TRAIN INPUT
        if Config.MODE == "CNN":
            if self.t_step < self.img_spike_train.shape[0]:
                self.configurationVector[self.input_neurons] += self.img_spike_train[self.t_step]
                if verbose:
                    print(f"Applied image spike train at step {self.t_step + 1}")
        
        elif self.single_spike_train.size > 0 and self.t_step < self.single_spike_train.shape[0]:
            if self.single_spike_train[self.t_step] == 1:
                self.configurationVector[self.input_neurons] += 1
                if verbose:
                    print(f"Applied spike train at step {self.t_step + 1}")
        
        extendedConfigVector = cp.zeros(cp.sum(self.ruleCountPerNeuron), dtype=cp.int32)
        idx = 0
        for i in range(len(self.configurationVector)):
            count = self.ruleCountPerNeuron[i]
            extendedConfigVector[idx:idx+count] = self.configurationVector[i]
            idx += count

        self.spikingVector = cp.ones_like(self.spikingVector) // (cp.ones_like(self.spikingVector) + cp.abs(extendedConfigVector - self.ruleVector))
        self.netGainVector = self.spikingVector @ self.sMpi
        self.configurationVector = self.configurationVector + self.netGainVector

        if Config.WHITE_HOLE:
            self.configurationVector = cp.zeros_like(self.configurationVector)

        return True

    def execute(self, verbose=False, startAgain=True):
        """Execute the system until halt condition is met"""
        if startAgain:
            self.t_step = 0
            
        if verbose:
            print("Initial Configuration Vector:", cp.asnumpy(self.configurationVector))
            print("-" * 30)
        
        # determine input length based on mode
        if Config.MODE == "CNN":
            input_length = self.img_spike_train.shape[0]
        else:
            input_length = len(self.single_spike_train) if isinstance(self.single_spike_train, (list, np.ndarray)) else 0
        
        while self.step(verbose=verbose) and (self.t_step < self.max_steps or self.t_step < input_length):
            if verbose:
                print("Step:", self.t_step + 1)
                print("Spiking Vector applied:", cp.asnumpy(self.spikingVector))
                print("Configuration Vector obtained:", cp.asnumpy(self.configurationVector))
                print("Net Gain Vector in step", self.t_step + 1, ":", cp.asnumpy(self.netGainVector))
                print("-" * 30)
            
            # Check halt condition
            if cp.all(self.spikingVector == 0) and (self.t_step >= input_length):
                print("Computation halts: spiking vector is zero, input is accepted")
                return True
            
            self.t_step += 1
        
        print("Computation halts: maximum number of steps reached, input is rejected")
        return False

    def rule_check(self, charge, source, div, mod, target):
        """Check if a rule can be applied based on neuron charge and rule parameters"""
        if charge > 0 and charge >= mod and charge >= target:
            if div > 0:
                return charge >= source and (charge - mod) % div == 0
            if div == 0:
                return charge >= source and charge == mod
        return False

    def get_configuration_vector(self):
        """Return configuration vector as NumPy array (copy from GPU)"""
        return cp.asnumpy(self.configurationVector)

    def get_spiking_vector(self):
        """Return spiking vector as NumPy array (copy from GPU)"""
        return cp.asnumpy(self.spikingVector)

    def get_net_gain_vector(self):
        """Return net gain vector as NumPy array (copy from GPU)"""
        return cp.asnumpy(self.netGainVector)

    def __str__(self):
        """String representation of the system state"""
        config_np = cp.asnumpy(self.configurationVector)
        spiking_np = cp.asnumpy(self.spikingVector)
        netgain_np = cp.asnumpy(self.netGainVector)
        transition_np = cp.asnumpy(self.spikingTransitionMatrix)
        
        return (f"Deterministic: {self.deterministic}\n"
                f"Spiking Transition Matrix:\n{transition_np}\n"
                f"Input Neurons: {self.input_neurons}\n"
                f"Configuration Vector: {config_np}\n"
                f"Spiking Vector: {spiking_np}\n"
                f"Net Gain Vector: {netgain_np}\n"
                f"Rule Vector: {self.ruleVector}\n"
                f"Target Vector: {target_np}")
