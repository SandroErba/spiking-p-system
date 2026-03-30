import numpy as np
import random
from sps.spike_utils import TransformationRule
from sps.snp_system import SNPSystem  
from sps.config import Config


class MSNPSystem:
    def __init__(self,configurationVector,spikingVector,spikingTransitionMatrix,netGainVector,ruleVector,max_steps=1000,deterministic=True,single_spike_train=None,input_neurons=None):
        self.configurationVector = configurationVector
        self.spikingVector = spikingVector
        self.spikingTransitionMatrix = spikingTransitionMatrix
        self.netGainVector = netGainVector
        self.ruleVector = ruleVector
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.applyingRuleVector = np.array([np.where(self.spikingTransitionMatrix[i] < 0)[0][0] for i in range(len(self.spikingTransitionMatrix))])
        
        self.targetVector = np.array([
            row[np.where(row > 0)[0][0]] if np.any(row > 0) else 0
            for row in self.spikingTransitionMatrix
        ])

        self.single_spike_train = single_spike_train
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

    def step(self):

        if Config.MODE == "CNN":
            if self.t_step < self.img_spike_train.shape[0]:
                self.configurationVector[self.input_neurons] += self.img_spike_train[self.t_step]

        elif self.single_spike_train and self.t_step < len(self.single_spike_train):
            if self.single_spike_train[self.t_step] == 1: # one boolean spike train for all the input neurons
                self.configurationVector[self.input_neurons] += 1

        self.update_spiking_vector()
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
        while self.step() and self.t_step < self.max_steps:
            if verbose:
                print("Step:", self.t_step + 1)
                print("Spiking Vector applied:", self.spikingVector)
                print("Configuration Vector obtained:", self.configurationVector)
                print("Net Gain Vector in step", self.t_step + 1, ":", self.netGainVector)
                print("-" * 30)
            if np.all(self.spikingVector == 0): # devo interrompere se ci sono ancora spike train da processare?
                print("Computation halts because the spiking vector is zero; no more rules can be applied; the input is accepted")
                return True
            self.t_step += 1
        print("Computation halts because the maximum number of steps has been reached; the input is rejected")
        return False

    # def check(self, charge):
    #     #with div and mod is possible to manage all value condition for charge
    #     if charge > 0 and charge >= self.mod and charge >= self.target: #for avoid negative values
    #         if self.div > 0:
    #             return charge >= self.source and (charge - self.mod) % self.div == 0
    #         if self.div == 0:
    #             return charge >= self.source and charge == self.mod
    #     return False

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
        


    def update_spiking_vector(self,verbose=False):
        for i in range(len(self.spikingVector)):
            self.spikingVector[i] = self.rule_check(self.configurationVector[self.applyingRuleVector[i]], abs(self.spikingTransitionMatrix[i][self.applyingRuleVector[i]]),self.ruleVector[0][i],self.ruleVector[1][i],self.targetVector[i])
        """
        # regex are in the form: a^mod (a^div)^*
        # with source == 0, the rule consumes all the spike
        """

        """
        # rule vector is in format:
        # ( (div1, div2, ..., divn),
        #   (mod1, mod2, ..., modn),
        )
        """

# da rivedere, perché faccio schifo
        if not self.deterministic:
            overlap = 0
            for i in range(1, len(self.spikingVector)):
                if self.spikingVector[i] == 1 and self.spikingVector[i-1] == 1 and self.applyingRuleVector[i] == self.applyingRuleVector[i-1]:
                   overlap += 1
                else:
                    if overlap > 0:
                        for j in range(i-overlap-1, i):
                            self.spikingVector[j] = 0
                        self.spikingVector[random.choice(range(i-overlap-1, i))] = 1
                        if verbose:
                            print("Non-deterministic choice made among", overlap + 1, "overlapping rules at neuron", self.applyingRuleVector[i])
                    overlap = 0

                
                   
