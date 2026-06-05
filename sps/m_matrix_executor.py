import numpy as np
from sps.spike_utils import TransformationRule  
from sps.m_snp_system import MSNPSystem  
from sps.snp_system import SNPSystem
from sps.config import Config
from sps.m_gpu import MSNPSystemGPU


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

                # Handle rule.mod - potrebbe essere lista o singolo valore
                if isinstance(rule.mod, list):
                    ruleVector[rule_idx] = rule.mod[0] if rule.mod else 0
                else:
                    ruleVector[rule_idx] = rule.mod
                    
                applyingRuleVector[rule_idx] = neuron.nid
                rule_idx += 1

        # Determina single_spike_train in base alla modalità
        single_spike_train = None
        if Config.MODE != "CNN":
            single_spike_train = SNPSystem.spike_train

        # Crea e ritorna l'istanza di MSNPSystemGPU (versione CuPy)
        # NOTA: NON passare synapsesMatrix due volte!
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