import numpy as np
from sps.spike_utils import TransformationRule  
from sps.m_snp_system import MSNPSystem  
from sps.snp_system import SNPSystem
from sps.config import Config

class MatrixExecutor:

    # This class is responsible for translating a SNPSystem from the sps.snp_system format to the MSNPSystem format
    def translate_to_matrix(self,SNPSystem):
        neurons = SNPSystem.neurons
        neurons_num = len(neurons)
        rule_num = sum(len(neuron.transf_rules) for neuron in neurons)

        deterministic = SNPSystem.deterministic
        max_steps = SNPSystem.max_steps

        # Initialize the vectors and matrices
        configurationVector = np.zeros(neurons_num, dtype=int)
        netGainVector = np.zeros((neurons_num,), dtype=int)
        spikingVector = np.zeros((rule_num,), dtype=int)
        spikingTransitionMatrix = np.zeros((rule_num, neurons_num), dtype=int) # as explained in the paper

        # I implemented these vectors in order to make the system executable
        targetVector = np.zeros((rule_num,), dtype=int) # how many spikes each rule produces, 0 for forgetting rules
        ruleVector = np.zeros((rule_num, 2), dtype=int)  # div and mod for each rule
        applyingRuleVector = np.zeros((rule_num,), dtype=int)  # which neuron each rule applies to

        rule_idx = 0
        input_neurons = [] # index of input neurons, to which the spike train will be applied

        for neuron in neurons:
            if neuron.neuron_type == 0:  # if it's an input neuron, add it to the list of input neurons
                input_neurons.append(neuron.nid)

            configurationVector[neuron.nid] = neuron.charge

            for rule in neuron.transf_rules:
                spikingTransitionMatrix[rule_idx, neuron.nid] = -rule.source

                for target in neuron.targets:
                    spikingTransitionMatrix[rule_idx, target] = rule.target 

                targetVector[rule_idx] = rule.target
                ruleVector[rule_idx] = [rule.div, rule.mod]
                applyingRuleVector[rule_idx] = neuron.nid
                rule_idx += 1

        return MSNPSystem(  configurationVector, 
                            spikingVector, 
                            spikingTransitionMatrix, 
                            netGainVector, 
                            ruleVector, 
                            max_steps, 
                            deterministic, 
                            single_spike_train = SNPSystem.spike_train if Config.MODE != "CNN" else None, 
                            input_neurons=input_neurons, 
                            targetVector=targetVector, 
                            applyingRuleVector=applyingRuleVector)


