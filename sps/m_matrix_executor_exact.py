import numpy as np
import torch
from sps.snp_system import SNPSystem
from sps.config import Config
from sps.m_snp_pytorch_exact_GPU_and_CPU import MSNPSystemExactGPU


class MatrixExecutor:

    # This class is responsible for translating a SNPSystem from the sps.snp_system format to the MSNPSystem format
    @staticmethod
    def translate_to_matrix(SNPSystem):
        neurons = SNPSystem.neurons
        neurons_num = len(neurons)
        rule_num = sum(len(neuron.transf_rules) for neuron in neurons)

        deterministic = SNPSystem.deterministic
        max_steps = SNPSystem.max_steps

        # Initialize the vectors and matrices
        configurationVector = np.zeros(neurons_num, dtype=int)
        spikingVector = np.zeros((rule_num,), dtype=int)
        spikingTransitionMatrix = np.zeros((rule_num, neurons_num), dtype=int)  # as explained in the paper
        synapsesMatrix = np.zeros((rule_num, neurons_num), dtype=int)      
        # I implemented these vectors in order to make the system executable
        ruleVector = np.zeros(rule_num, dtype=int)  # exact E goes in mod, div is for all rules in this implementation
        applyingRuleVector = np.zeros((rule_num,), dtype=int)  # which neuron each rule applies to

        rule_idx = 0
        input_neurons = []  # index of input neurons, to which the spike train will be applied
        output_neurons = []  # index of output neurons, to which the output will be read from

        for neuron in neurons:
            if neuron.neuron_type == 0:  # if it's an input neuron, add it to the list of input neurons
                input_neurons.append(neuron.nid)
            elif neuron.neuron_type == 2:  # if it's an output neuron, add it to the list of output neurons
                output_neurons.append(neuron.nid)

            configurationVector[neuron.nid] = neuron.charge

            for rule in neuron.transf_rules:
                spikingTransitionMatrix[rule_idx, neuron.nid] = -rule.source
                synapsesMatrix[rule_idx, neuron.nid] = 1
                
                for target in neuron.targets:
                    spikingTransitionMatrix[rule_idx, target] = rule.target 
                    synapsesMatrix[rule_idx, target] = 1 if target > 0 else -1

                ruleVector[rule_idx] = rule.mod  # exact goes in mod - assume div = 0
                    
                applyingRuleVector[rule_idx] = neuron.nid
                rule_idx += 1

        # Determina single_spike_train in base alla modalità
        single_spike_train = None
        if Config.MODE != "CNN":
            single_spike_train = SNPSystem.spike_train

        # Crea e ritorna l'istanza di MSNPSystemExactGPU (versione PyTorch)
        return MSNPSystemExactGPU(
            configurationVector=configurationVector,
            spikingVector=spikingVector,
            spikingTransitionMatrix=spikingTransitionMatrix,
            synapsesMatrix=synapsesMatrix,
            ruleVector=ruleVector,
            max_steps=max_steps,
            deterministic=deterministic,
            single_spike_train=single_spike_train,
            input_neurons=input_neurons,
            output_neurons=output_neurons,
            applyingRuleVector=applyingRuleVector,
            device='gpu',
            testsize = SNPSystem.input_len
        )