import numpy as np
from sps.spike_utils import TransformationRule  
from sps.m_snp_system import MSNPSystem  
from sps.snp_system import SNPSystem
from sps.config import Config
from sps.m_gpu import MSNPSystemGPU


class MatrixExecutor:

    @staticmethod
    def translate_to_matrix(SNPSystem):
        neurons = SNPSystem.neurons
        neurons_num = len(neurons)
        rule_num = sum(len(neuron.transf_rules) for neuron in neurons)

        deterministic = SNPSystem.deterministic
        max_steps = SNPSystem.max_steps

        configurationVector = np.zeros(neurons_num, dtype=int)
        spikingVector = np.zeros((rule_num,), dtype=int)
        spikingTransitionMatrix = np.zeros((rule_num, neurons_num), dtype=int)
        synapsesMatrix = np.zeros((rule_num, neurons_num), dtype=int)      
        ruleVector = np.zeros(rule_num, dtype=int)
        applyingRuleVector = np.zeros((rule_num,), dtype=int)

        rule_idx = 0
        input_neurons = []

        for neuron in neurons:
            if neuron.neuron_type == 0:
                input_neurons.append(neuron.nid)

            configurationVector[neuron.nid] = neuron.charge

            for rule in neuron.transf_rules:
                spikingTransitionMatrix[rule_idx, neuron.nid] = -rule.source
                synapsesMatrix[rule_idx, neuron.nid] = 1
                
                for target in neuron.targets:
                    spikingTransitionMatrix[rule_idx, target] = rule.target 
                    synapsesMatrix[rule_idx, target] = 1 if target > 0 else -1

                ruleVector[rule_idx] = rule.mod
                    
                applyingRuleVector[rule_idx] = neuron.nid
                rule_idx += 1

        single_spike_train = None
        if Config.MODE != "CNN":
            single_spike_train = SNPSystem.spike_train

        return MSNPSystemGPU(
            configurationVector=configurationVector,
            spikingVector=spikingVector,
            spikingTransitionMatrix=spikingTransitionMatrix,
            synapsesMatrix=synapsesMatrix,
            ruleVector=ruleVector,
            max_steps=max_steps,
            deterministic=deterministic,
            single_spike_train=single_spike_train,
            input_neurons=input_neurons,
            applyingRuleVector=applyingRuleVector
        )