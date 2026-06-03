import numpy as np
import random
import cupy as cp
from sps.spike_utils import TransformationRule
from sps.snp_system import SNPSystem  
from sps.config import Config


class MSNPSystemGPU:
    """
    GPU-accelerated version of MSNPSystem using CuPy for matrix operations.
    All arrays are stored on GPU and operations are computed on GPU.
    """
    
    def __init__(self, configurationVector, spikingVector, spikingTransitionMatrix, 
                 netGainVector, ruleVector, max_steps=1000, deterministic=True, 
                 single_spike_train=None, input_neurons=None, targetVector=None, 
                 applyingRuleVector=None):
        
        if configurationVector is None or spikingTransitionMatrix is None or ruleVector is None:
            raise ValueError("configurationVector, spikingTransitionMatrix and ruleVector cannot be None")
        
        rule_num = len(spikingTransitionMatrix)
        neuron_num = len(configurationVector)

        # Convert to CuPy arrays (automatically on GPU)
        self.configurationVector = cp.asarray(configurationVector, dtype=cp.int32)
        self.spikingTransitionMatrix = cp.asarray(spikingTransitionMatrix, dtype=cp.int32)
        
        if spikingVector is None:
            self.spikingVector = cp.zeros(rule_num, dtype=cp.int32)
        else:
            self.spikingVector = cp.asarray(spikingVector, dtype=cp.int32)

        if netGainVector is None:
            self.netGainVector = cp.zeros(neuron_num, dtype=cp.int32)
        else:
            self.netGainVector = cp.asarray(netGainVector, dtype=cp.int32)

        """
        rule vector is in format
        ((div1,mod1),
        (div2,mod2),
        ...
        (divn,modn))
        """
        self.ruleVector = ruleVector

        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        self.max_steps = max_steps
        self.deterministic = deterministic

        # add antispike check
        # each entry is the index of the neuron to which the rule applies
        if applyingRuleVector is None:
            raise ValueError("ApplyingRuleVector is required.")
        self.applyingRuleVector = cp.asarray(applyingRuleVector, dtype=cp.int32)

        # dictionary that maps each neuron to the list of rules that apply to it
        self.rulePerNeuron = {i: [] for i in range(neuron_num)} 
        for i in range(rule_num):
            neuron = int(self.applyingRuleVector[i].item())
            self.rulePerNeuron[neuron].append(i)

        # Target Vector: monodimensional vector of length n
        if targetVector is not None:
            self.targetVector = cp.asarray(targetVector, dtype=cp.int32)
        else:
            raise ValueError("targetVector is required.")

        self.ruleVector = cp.asarray(ruleVector, dtype=cp.int32)

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

        # UPDATE SPIKING VECTOR
        self.update_spiking_vector(verbose=verbose)

        # GPU ACCELERATED MATRIX-VECTOR MULTIPLICATION
        # netGainVector = spikingVector @ spikingTransitionMatrix
        self.netGainVector = self.spikingVector @ self.spikingTransitionMatrix
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

    def update_spiking_vector(self, verbose=False):
        """Update spiking vector based on rule applicability"""
        rule_num = self.spikingTransitionMatrix.shape[0]

        # Gather charge and source values entirely on GPU
        charges = self.configurationVector[self.applyingRuleVector]
        row_indices = cp.arange(rule_num, dtype=cp.int32)
        sources = cp.abs(self.spikingTransitionMatrix[row_indices, self.applyingRuleVector])
        divs = self.ruleVector[:, 0]
        mods = self.ruleVector[:, 1]
        targets = self.targetVector

        valid_charge = (charges > 0) & (charges >= mods) & (charges >= targets)
        fire_div = (divs > 0) & (charges >= sources) & ((charges - mods) % divs == 0)
        fire_zero = (divs == 0) & (charges >= sources) & (charges == mods)

        self.spikingVector = (valid_charge & (fire_div | fire_zero)).astype(cp.int32)

        if not self.deterministic:
            neuron_rule_map = {}
            spiking_indices = cp.where(self.spikingVector == 1)[0]
            for i in spiking_indices.tolist():
                neuron = int(self.applyingRuleVector[i].item())
                neuron_rule_map.setdefault(neuron, []).append(i)

            for neuron, rules in neuron_rule_map.items():
                if len(rules) > 1:
                    chosen_rule = random.choice(rules)
                    for rule in rules:
                        if rule != chosen_rule:
                            self.spikingVector[rule] = 0
                    if verbose:
                        print(f"Non-deterministic choice at neuron {neuron}: "
                              f"selected rule {chosen_rule}")

    def elem_add(self, a, b):
        """Element-wise addition on GPU"""
        a = cp.asarray(a, dtype=cp.int32)
        b = cp.asarray(b, dtype=cp.int32)
        return a + b

    def elem_subtract(self, a, b):
        """Element-wise subtraction on GPU"""
        a = cp.asarray(a, dtype=cp.int32)
        b = cp.asarray(b, dtype=cp.int32)
        return a - b

    def elem_multiply(self, a, b):
        """Element-wise multiplication (Hadamard product) on GPU"""
        a = cp.asarray(a, dtype=cp.int32)
        b = cp.asarray(b, dtype=cp.int32)
        return a * b

    def elem_divide(self, a, b):
        """Element-wise division on GPU (integer division, 0 for division by zero)"""
        a = cp.asarray(a, dtype=cp.int32)
        b = cp.asarray(b, dtype=cp.int32)
        result = cp.zeros_like(a)
        mask = b != 0
        result[mask] = a[mask] // b[mask]
        return result

    def elem_modulo(self, a, b):
        """Element-wise modulo on GPU (0 for modulo by zero)"""
        a = cp.asarray(a, dtype=cp.int32)
        b = cp.asarray(b, dtype=cp.int32)
        result = cp.zeros_like(a)
        mask = b != 0
        result[mask] = a[mask] % b[mask]
        return result

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
        target_np = cp.asnumpy(self.targetVector)
        
        return (f"Deterministic: {self.deterministic}\n"
                f"Spiking Transition Matrix:\n{transition_np}\n"
                f"Input Neurons: {self.input_neurons}\n"
                f"Configuration Vector: {config_np}\n"
                f"Spiking Vector: {spiking_np}\n"
                f"Net Gain Vector: {netgain_np}\n"
                f"Rule Vector: {self.ruleVector}\n"
                f"Target Vector: {target_np}")
