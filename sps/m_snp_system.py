import numpy as np
import random
from sps.spike_utils import TransformationRule  # Changed back to absolute import
from sps.snp_system import SNPSystem  # Changed back to absolute import

class MSNPSystem:
    def __init__(self,configurationVector,spikingVector,spikingTransitionMatrix,netGainVector,ruleVector,max_steps=1000,deterministic=True):
        self.configurationVector = configurationVector
        self.spikingVector = spikingVector
        self.spikingTransitionMatrix = spikingTransitionMatrix
        self.netGainVector = netGainVector
        self.ruleVector = ruleVector
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.applyingRuleVector = np.array([np.where(self.spikingTransitionMatrix[i] < 0)[0][0] for i in range(len(self.spikingTransitionMatrix))])
  

    def step(self):
        if self.max_steps <= 0:
            return False

        self.update_spiking_vector()
        self.configurationVector = self.configurationVector + self.spikingVector @ self.spikingTransitionMatrix
        self.netGainVector = self.spikingVector @ self.spikingTransitionMatrix

        self.max_steps -= 1
        return True
    
    def execute(self,verbose=False):
        step = 0
        if verbose:
            print("Initial Configuration Vector:", self.configurationVector)
            print("-" * 30)
        while self.step():
            if verbose:
                print("Step:", step + 1)
                print("Spiking Vector applied:", self.spikingVector)
                print("Configuration Vector obtained:", self.configurationVector)
                print("Net Gain Vector in step", step + 1, ":", self.netGainVector)
                print("-" * 30)
            if np.all(self.spikingVector == 0):
                print("Computation halts because the spiking vector is zero; no more rules can be applied; the input is accepted")
                return True
            step += 1
        print("Computation halts because the maximum number of steps has been reached; the input is rejected")
        return False


    def update_spiking_vector(self,verbose=False):
        for i in range(len(self.spikingVector)):
            self.spikingVector[i] = self.ruleVector[i].check(self.configurationVector[self.applyingRuleVector[i]])

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

                
                   
