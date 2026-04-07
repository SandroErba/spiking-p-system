import numpy as np
from sps.spike_utils import TransformationRule  
from sps.m_snp_system import MSNPSystem  
from sps.snp_system import SNPSystem

class MatrixExecutor:

    def translate_to_matrix(self,SNPSystem):
        neurons = SNPSystem.neurons
        deterministic = SNPSystem.deterministic
        max_steps = SNPSystem.max_steps
        configurationVector = np.zeros(len(neurons), dtype=int)
        netGainVector = np.zeros((len(neurons),), dtype=int)
        rules = [rule for neuron in neurons for rule in neuron.transf_rules]
        spikingVector = np.zeros((len(rules),), dtype=int)
        spikingTransitionMatrix = np.zeros((len(rules), len(neurons)), dtype=int)
        targetVector = np.zeros((len(rules),), dtype=int)
        ruleVector = np.zeros((len(rules), 2), dtype=int)  # div e mod per ogni regola
        applyingRuleVector = np.zeros((len(rules),), dtype=int)  
        rule_idx = 0
        input_neurons = []
        for neuron in neurons:
            if neuron.neuron_type == 0:  # se è un neurone di input, aggiungilo alla lista degli input neurons
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

        return MSNPSystem(configurationVector, spikingVector, spikingTransitionMatrix, netGainVector, ruleVector, max_steps, deterministic, single_spike_train = SNPSystem.spike_train, input_neurons=input_neurons, targetVector=targetVector, applyingRuleVector=applyingRuleVector)


