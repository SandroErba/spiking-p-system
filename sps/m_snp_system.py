class MSNPSystem:
    def __init__(self,configurationVector,spikingVector,spikingTransitionMatrix,netGainVector,thresholdVector):
        self.configurationVector = configurationVector
        self.spikingVector = spikingVector
        self.spikingTransitionMatrix = spikingTransitionMatrix
        self.netGainVector = netGainVector
        self.thresholdVector = thresholdVector   
    