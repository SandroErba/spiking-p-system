import numpy as np
import random
from sps.spike_utils import TransformationRule
from sps.snp_system import SNPSystem  
from sps.config import Config


class MSNPSystem:
    def __init__(self,configurationVector,spikingVector,spikingTransitionMatrix,netGainVector,ruleVector,max_steps=1000,deterministic=True,single_spike_train=None,input_neurons=None,targetVector=None, applyingRuleVector=None):
        self.configurationVector = configurationVector
        self.spikingVector = spikingVector
        self.spikingTransitionMatrix = spikingTransitionMatrix
        self.netGainVector = netGainVector
        self.ruleVector = ruleVector
        self.max_steps = max_steps
        self.deterministic = deterministic
        if applyingRuleVector is None:
            self.applyingRuleVector = np.array([np.where(self.spikingTransitionMatrix[i] < 0)[0][0] for i in range(len(self.spikingTransitionMatrix))])
        else:
            self.applyingRuleVector = applyingRuleVector

        """
        il target vector è un vettore monodimensionale di lunghezza n 
        possiede il numero di spike che ogni regola produce, se è una firing rule, 
        o 0 se è una forgetting rule, e lo prendo dalla spikingTransitionMatrix
        """
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
        self.t_step = 0

        """
        le immagini in input, che sono (per ora) un array monodimensionale,
        sono nel formato 1xn, dove n è il risultato di altezza x larghezza dell'immagine
        quindi come input passerò un array di dimensione (m,n) dove m è il numero di immagini 
        da processare, e n è il numero di pixel (o input neurons) per ogni immagine
        """

    def loadImages(self,img_spike_train):
        self.img_spike_train = img_spike_train

    def step(self,verbose=False):

        # INPUT DA SPIKE TRAIN (img/boolean)
        if Config.MODE == "CNN":
            if self.t_step < self.img_spike_train.shape[0]:
                self.configurationVector[self.input_neurons] += self.img_spike_train[self.t_step]

        elif self.single_spike_train and self.t_step < len(self.single_spike_train):
            if self.single_spike_train[self.t_step] == 1: # one boolean spike train for all the input neurons
                self.configurationVector[self.input_neurons] += 1
                if verbose:
                    print(f"Applied spike train at step {self.t_step + 1}: added 1 spike to input neurons {self.input_neurons}")

        # UPDATE SPIKING VECTOR
        self.update_spiking_vector()

        # CALCOLO STEP -> UPDATE CONFIGURATION VECTOR AND NET GAIN VECTOR
        self.configurationVector = self.configurationVector + self.spikingVector @ self.spikingTransitionMatrix
        self.netGainVector = self.spikingVector @ self.spikingTransitionMatrix

        if Config.WHITE_HOLE:
            self.configurationVector = np.zeros_like(self.configurationVector)

        return True
    
    def execute(self,verbose=False):
        self.t_step = 0
        if verbose:
            print("Initial Configuration Vector:", self.configurationVector)
            print("-" * 30)
        while self.step(verbose=verbose) and (self.t_step < self.max_steps or self.t_step < len(self.single_spike_train)):
            if verbose:
                print("Step:", self.t_step + 1)
                print("Spiking Vector applied:", self.spikingVector)
                print("Configuration Vector obtained:", self.configurationVector)
                print("Net Gain Vector in step", self.t_step + 1, ":", self.netGainVector)
                print("-" * 30)
            if np.all(self.spikingVector == 0) and (self.t_step >= len(self.single_spike_train)): # devo interrompere se ci sono ancora spike train da processare?
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

        
        # regex are in the form: a^mod (a^div)^*
    
        """
        rule vector is in format
        ((div1,mod1),
        (div2,mod2),
        ...
        (divn,modn))
         """

    def update_spiking_vector(self, verbose=False):
        if not self.deterministic:
            neuron_rule_map = {}  # chiave: indice neurone, valore: set di indici regole
        
        for i in range(len(self.spikingVector)):
            self.spikingVector[i] = self.rule_check(
                self.configurationVector[self.applyingRuleVector[i]], 
                abs(self.spikingTransitionMatrix[i][self.applyingRuleVector[i]]),
                self.ruleVector[i][0], 
                self.ruleVector[i][1], 
                self.targetVector[i]
            )
            
            if not self.deterministic and self.spikingVector[i] == 1:
                neuron = self.applyingRuleVector[i]
                if neuron not in neuron_rule_map:
                    neuron_rule_map[neuron] = set()
                neuron_rule_map[neuron].add(i)
        
        if not self.deterministic and neuron_rule_map:
            for neuron, rules in neuron_rule_map.items():
                if len(rules) > 1:  # più regole attive per lo stesso neurone
                    # Scegli una regola casuale tra quelle attive
                    chosen_rule = random.choice(list(rules))
                    # Disattiva tutte le altre regole per questo neurone
                    for rule in rules:
                        if rule != chosen_rule:
                            self.spikingVector[rule] = 0
                    if verbose:
                        print(f"Non-deterministic choice at neuron {neuron}: "
                            f"selected rule {chosen_rule} among {rules}")

    def __str__(self):
        return f"Deterministic: {self.deterministic}\nSpikingTransitionMatrix: {self.spikingTransitionMatrix}\nInputNeurons: {self.input_neurons}\nConfiguration Vector: {self.configurationVector}\nSpiking Vector: {self.spikingVector}\nNet Gain Vector: {self.netGainVector}\nRule Vector: {self.ruleVector}\nTarget Vector: {self.targetVector}"

                
                   
