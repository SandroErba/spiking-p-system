import numpy as np
import random
from sps.spike_utils import TransformationRule
from sps.snp_system import SNPSystem  
from sps.config import Config


class MSNPSystem:
    def __init__(self,configurationVector,spikingVector,spikingTransitionMatrix,netGainVector,ruleVector,max_steps=1000,deterministic=True,single_spike_train=None,input_neurons=None,targetVector=None, applyingRuleVector=None):
        if configurationVector is None or spikingTransitionMatrix is None or ruleVector is None:
            raise ValueError("configurationVector, spikingTransitionMatrix and ruleVector cannot be None")
        
        rule_num = len(spikingTransitionMatrix)
        neuron_num = len(configurationVector)

        self.configurationVector = configurationVector
        if spikingVector is None:
            self.spikingVector = np.zeros(rule_num, dtype=int)
        else:
            self.spikingVector = spikingVector

        self.spikingTransitionMatrix = spikingTransitionMatrix

        if netGainVector is None:
            self.netGainVector = np.zeros(neuron_num, dtype=int)
        else:
            self.netGainVector = netGainVector

        # regex are in the form: a^mod (a^div)^*
    
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
            self.applyingRuleVector = np.array([np.where(self.spikingTransitionMatrix[i] < 0)[0][0] for i in range(rule_num)])
        else:
            self.applyingRuleVector = applyingRuleVector

        # dictionary that maps each neuron to the list of rules that apply to it
        # used for efficiently checking the deterministic condition that at most one rule can apply to each neuron
        self.rulePerNeuron = {i: [] for i in range(neuron_num)} 
        for i in range(rule_num):
            neuron = self.applyingRuleVector[i]
            self.rulePerNeuron[neuron].append(i)

        # Target Vector: monodimensional vector of length n
        # where each entry is the number of spikes produced by the rule if it's a firing rule,
        #  or 0 for forgetting rules, taken from the spikingTransitionMatrix
        # !!! I assume that for each rule, the target is the same for all the neurons it applies to !!!
        if targetVector is not None:
            self.targetVector = targetVector
        else:
            self.targetVector = np.array([
                row[np.where(row > 0)[0][0]] if np.any(row > 0) else 0
                for row in self.spikingTransitionMatrix
            ])

        if single_spike_train is not None:
            self.single_spike_train = single_spike_train
        else:
            self.single_spike_train = []

        if input_neurons is None:
            self.input_neurons = []
        else:
            self.input_neurons = input_neurons
        
        # Initialize the time step counter
        self.t_step = 0


    # Input images as spike trains, in the format (num_images, num_input_neurons)
    # For example, for 28x28 images, num_input_neurons would be 784 
    # and each image would be represented as a vector of length 784
    def loadImages(self,img_spike_train):
        # Reshape da (N, 28, 28) a (N, 784) if necessary
        if len(img_spike_train.shape) == 3:
            self.img_spike_train = img_spike_train.reshape(img_spike_train.shape[0], -1)
        else:
            self.img_spike_train = img_spike_train


    def step(self,verbose=False):

        # SPIKE TRAIN INPUT
        # CNN -> Images
        if Config.MODE == "CNN":
            if self.t_step < self.img_spike_train.shape[0]:
                self.configurationVector[self.input_neurons] += self.img_spike_train[self.t_step]
                if verbose:
                    print(f"Applied image spike train at step {self.t_step + 1}: added {self.img_spike_train[self.t_step]} spikes to input neurons {self.input_neurons}")
        
        # SINGLE SPIKE TRAIN -> boolean spike train for all input neurons
        elif isinstance(self.single_spike_train, (list, np.ndarray)) and len(self.single_spike_train) > 0 and self.t_step < len(self.single_spike_train):
            if self.single_spike_train[self.t_step] == 1: # one boolean spike train for all the input neurons
                self.configurationVector[self.input_neurons] += 1
                if verbose:
                    print(f"Applied spike train at step {self.t_step + 1}: added 1 spike to input neurons {self.input_neurons}")

        # UPDATE SPIKING VECTOR
        self.update_spiking_vector()

        # CRUCIAL STEP -> UPDATE CONFIGURATION VECTOR AND NET GAIN VECTOR
        self.netGainVector = self.spikingVector @ self.spikingTransitionMatrix
        self.configurationVector = self.configurationVector + self.netGainVector

        if Config.WHITE_HOLE:
            self.configurationVector = np.zeros_like(self.configurationVector)

        return True
    

    def execute(self,verbose=False,startAgain=True):
        if startAgain:
            self.t_step = 0
            
        if verbose:
            print("Initial Configuration Vector:", self.configurationVector)
            print("-" * 30)
        
        # determine input length based on mode and available spike trains
        if Config.MODE == "CNN":
            input_length = self.img_spike_train.shape[0]
        else:
            input_length = len(self.single_spike_train) if isinstance(self.single_spike_train, (list, np.ndarray)) else 0
        
        while self.step(verbose=verbose) and (self.t_step < self.max_steps or self.t_step < input_length):
            if verbose:
                print("Step:", self.t_step + 1)
                print("Spiking Vector applied:", self.spikingVector)
                print("Configuration Vector obtained:", self.configurationVector)
                print("Net Gain Vector in step", self.t_step + 1, ":", self.netGainVector)
                print("-" * 30)
            if np.all(self.spikingVector == 0) and (self.t_step >= input_length):
                print("Computation halts because the spiking vector is zero; no more rules can be applied; the input is accepted")
                return True
            self.t_step += 1
        print("Computation halts because the maximum number of steps has been reached; the input is rejected")
        return False



    """
    - charge: the current charge of the neuron
    - source: the number of spikes consumed by the rule
    - div: the divisor in the rule's regular expression (a^div)*
    - mod: the modulus in the rule's regular expression (a^mod)
    - target: the number of spikes produced by the rule (if > 0) or 0 for forgetting rules
    """
    def rule_check(self, charge, source, div, mod,target):
        if charge > 0 and charge >= mod and charge >= target:
            if div > 0:
                return charge >= source and (charge - mod) % div == 0
            if div == 0:
                return charge >= source and charge == mod
        return False



    def update_spiking_vector(self, verbose=False):
        if not self.deterministic:
            neuron_rule_map = {}  # key: neuron index, value: set of rule indices that want to fire for this neuron
        
        idxs = [ i for i in range(rule_num)]

        # da testare
        for i in idxs:
            self.spikingVector[i] = self.rule_check(
                self.configurationVector[self.applyingRuleVector[i]], 
                abs(self.spikingTransitionMatrix[i][self.applyingRuleVector[i]]),
                self.ruleVector[i][0], 
                self.ruleVector[i][1], 
                self.targetVector[i]
            )

            if self.spikingVector[i] == 1:
                if deterministic: # skip the check for multiple rules applying to the same neuron
                    neuron = self.applyingRuleVector[i]
                    self.spikingVector[self.rulePerNeuron[neuron]] = 0
                    self.spikingVector[i] = 1
                    idxs.remove(self.rulePerNeuron[neuron])

                else: #nondeterministic case - build the neuron_rule_map to resolve conflicts later
                    neuron = self.applyingRuleVector[i]
                    if neuron not in neuron_rule_map:
                        neuron_rule_map[neuron] = set()
                    neuron_rule_map[neuron].add(i)
        
        # In the non-deterministic case, resolve conflicts by randomly choosing 
        # one rule for each neuron that has multiple applicable rules

        if not self.deterministic and neuron_rule_map:
            for neuron, rules in neuron_rule_map.items():
                if len(rules) > 1:  # more than one rule applyable for this neuron
                    # Choose one rule randomly
                    chosen_rule = random.choice(list(rules))
                    # deactivate all other rules for this neuron
                    for rule in rules:
                        if rule != chosen_rule:
                            self.spikingVector[rule] = 0
                    if verbose:
                        print(f"Non-deterministic choice at neuron {neuron}: "
                            f"selected rule {chosen_rule} among {rules}")

    def __str__(self):
        return f"Deterministic: {self.deterministic}\nSpikingTransitionMatrix: {self.spikingTransitionMatrix}\nInputNeurons: {self.input_neurons}\nConfiguration Vector: {self.configurationVector}\nSpiking Vector: {self.spikingVector}\nNet Gain Vector: {self.netGainVector}\nRule Vector: {self.ruleVector}\nTarget Vector: {self.targetVector}"

                
                   
