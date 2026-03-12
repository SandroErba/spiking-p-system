import numpy as np
from spike_utils import TransformationRule

class MatrixExecutor:


    # Da capire per gil spike train
    def __init__(self, SNPSystem):
        self.SNPSystem = SNPSystem
        self.neurons = SNPSystem.neurons
        self.deterministic = SNPSystem.deterministic
        self.max_steps = SNPSystem.max_steps

    # Necessito di un:
    # - configuration vector
    # - spiking vector
    # - spiking transition matrix
    # - net gain vector

    @staticmethod
    def test():
        # Let's test the example I made
        c0 = np.array([9,0,0])
        r1 = TransformationRule(1,2,2,2,0)
        r2 = TransformationRule(0,1,1,1,0)
        r3 = TransformationRule(0,2,2,1,0)
        r4 = TransformationRule(0,1,1,1,0)
        rules = [r1,r2,r3,r4]
        appliedNeurons = [0,1,1,2]
        spikingVector = np.zeros((len(rules),), dtype=int)
        for i in range(len(rules)):
            spikingVector[i] = rules[i].check(c0[appliedNeurons[i]])
        print(spikingVector)
        print(spikingVector.shape)



        spikingTransitionMatrix = np.array([
            [-2,2,0],
            [0,-1,1],
            [0,-2,1],
            [1,0,-1]

        ])

        print(spikingTransitionMatrix.shape)
        c1 = c0 + spikingVector @ spikingTransitionMatrix
        print(c1)

    def __str__(self):
        output = ""
        for neuron in self.neurons:
            output += neuron.__str__() + "\n"
        return output

if __name__ == "__main__":
    MatrixExecutor.test()
