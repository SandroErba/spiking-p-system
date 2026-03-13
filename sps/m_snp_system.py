class MSNPSystem:
    def __init__(self,configurationVector,spikingVector,spikingTransitionMatrix,netGainVector,max_steps=1000,deterministic=True):
        self.configurationVector = configurationVector
        self.spikingVector = spikingVector
        self.spikingTransitionMatrix = spikingTransitionMatrix
        self.netGainVector = netGainVector
        self.max_steps = max_steps
        self.deterministic = deterministic
  

    def step(self):
        if self.max_steps <= 0:
            return False
        self.configurationVector = self.configurationVector + self.spikingVector @ self.spikingTransitionMatrix
        # netgain
        self.max_steps -= 1
        return True
        
    def update_spiking_vector(self):
        for i in range(len(self.spikingVector)):
            self.spikingVector[i] = 1 if self.configurationVector[i] >= self.netGainVector[i] else 0
    