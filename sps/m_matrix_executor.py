import numpy as np
from sps.spike_utils import TransformationRule  # Fixed import
from sps.m_snp_system import MSNPSystem  # Fixed import
from sps.snp_system import SNPSystem  # Fixed import

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
        #for i in range(len(rules)):
        # #    spikingVector[i] = rules[i].check(c0[appliedNeurons[i]])
        # print(spikingVector)
        # print(spikingVector.shape)

        # for i in range(len(spikingVector)):
        #     spikingVector[i] = rules[i].check(c0[np.where(spikingTransitionMatrix[i] < 0)[0][0]])
        # print("test", spikingVector)

        # print(spikingTransitionMatrix.shape)
        # c1 = c0 + spikingVector @ spikingTransitionMatrix
        # print(c1)
        c0 = np.array([9,0,0])
        r1 = TransformationRule(1,2,2,2,0)
        r2 = TransformationRule(0,1,1,1,0)
        r3 = TransformationRule(0,2,2,1,0)
        r4 = TransformationRule(0,1,1,1,0)
        rules = [r1,r2,r3,r4]
        spikingVector = np.zeros((len(rules),), dtype=int)
        netGainVector = np.zeros((len(rules),), dtype=int)
        spikingTransitionMatrix = np.array([
            [-2,2,0],
            [0,-1,1],
            [0,-2,1],
            [1,0,-1]
        ])

        test1 = MSNPSystem(c0,spikingVector,spikingTransitionMatrix,netGainVector,rules)
        print(test1.applyingRuleVector)
        test1.execute(True)
        print(test1.configurationVector)

    @staticmethod
    def translate_to_matrix(SNPSystem):
        neurons = SNPSystem.neurons
        deterministic = SNPSystem.deterministic
        max_steps = SNPSystem.max_steps
        configurationVector = np.zeros(len(neurons), dtype=int)
        spikingVector = np.zeros((len(neurons),), dtype=int)
        netGainVector = np.zeros((len(neurons),), dtype=int)
        rules = []
        for neuron in neurons:
            configurationVector[neuron.nid] = neuron.charge
            for rule in neuron.transf_rules:
                rules.append(rule)
        # Fix: Initialize matrix after rules are collected
        spikingTransitionMatrix = np.zeros((len(rules), len(neurons)), dtype=int)
        rule_idx = 0
        for neuron in neurons:
            for rule in neuron.transf_rules:
                spikingTransitionMatrix[rule_idx, neuron.nid] = -rule.source
                for target in neuron.targets:
                    spikingTransitionMatrix[rule_idx, target] += rule.target  # Accumulate if multiple targets
                rule_idx += 1
        return MSNPSystem(configurationVector, spikingVector, spikingTransitionMatrix, netGainVector, rules)
        

    def __str__(self):
        output = ""
        for neuron in self.neurons:
            output += neuron.__str__() + "\n"
        return output

    @staticmethod
    def test2():
        snps = SNPSystem(0, 100, True)  # Fix: Pass 0 for input_len to avoid None error
        # Fix: Update path for root execution
        snps.load_neurons_from_csv("sps/csv/prova.csv")
        translatedSystem = MatrixExecutor.translate_to_matrix(snps)
        print(translatedSystem.spikingTransitionMatrix)

if __name__ == "__main__":
    #snps = SNPSystem(0, 100, True)  # Fix: Pass 0 for input_len to avoid None error
    # Fix: Update path for root execution
    #snps.load_neurons_from_csv("csv/prova.csv")
    #translatedSystem = MatrixExecutor.translate_to_matrix(snps)
    #print(translatedSystem.spikingTransitionMatrix)
    snps = SNPSystem(0, 10, False)  # Fix: Pass 0 for input_len to avoid None error
    snps.load_neurons_from_csv("csv/" + "ExampleExtended.csv")
    translatedSystem = MatrixExecutor.translate_to_matrix(snps)
    print(translatedSystem.spikingTransitionMatrix)

