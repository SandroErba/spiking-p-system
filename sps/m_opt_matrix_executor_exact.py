# File: sps/m_opt_matrix_executor_exact.py

import numpy as np
import torch
from sps.snp_system import SNPSystem
from sps.config import Config
from sps.m_opt_snp_pytorch_exact_GPU_and_CPU import MSNPSystemExactGPU


class MatrixExecutor:
    
    @staticmethod
    def translate_to_matrix(snp_system, device="cpu"):
        """
        Traduzione da SNPSystem a MSNPSystemExactGPU.
        Decide automaticamente se usare formato sparse o dense.
        """
        neurons = snp_system.neurons
        neurons_num = len(neurons)
        
        # Conta regole
        rule_num = sum(len(neuron.transf_rules) for neuron in neurons)
        
        # Allocazione vettori NumPy (sempre densi, piccoli)
        configurationVector = np.zeros(neurons_num, dtype=np.int32)
        spikingVector = np.zeros(rule_num, dtype=np.int32)
        ruleVector = np.zeros(rule_num, dtype=np.int32)
        applyingRuleVector = np.zeros(rule_num, dtype=np.int32)
        
        # Liste per costruzione efficiente
        stm_rows, stm_cols, stm_data = [], [], []
        sm_rows, sm_cols, sm_data = [], [], []
        
        input_neurons = []
        output_neurons = []
        rule_idx = 0
        
        # Costruzione matrici
        for neuron in neurons:
            nid = neuron.nid
            neuron_type = neuron.neuron_type
            
            if neuron_type == 0:
                input_neurons.append(nid)
            elif neuron_type == 2:
                output_neurons.append(nid)
            
            configurationVector[nid] = neuron.charge
            
            for rule in neuron.transf_rules:
                # Self-connection
                stm_rows.append(rule_idx)
                stm_cols.append(nid)
                stm_data.append(-rule.source)
                
                sm_rows.append(rule_idx)
                sm_cols.append(nid)
                sm_data.append(1)
                
                # Target connections
                target_value = rule.target
                if target_value != 0:
                    for target in neuron.targets:
                        abs_target = abs(target)
                        
                        stm_rows.append(rule_idx)
                        stm_cols.append(abs_target)
                        stm_data.append(target_value)
                        
                        sm_rows.append(rule_idx)
                        sm_cols.append(abs_target)
                        sm_data.append(1 if target > 0 else -1)
                
                ruleVector[rule_idx] = rule.mod
                applyingRuleVector[rule_idx] = nid
                rule_idx += 1
        
        # Determina se usare sparse (conveniente se < 20% densità)
        nonzeros = len(stm_rows)
        total_elements = rule_num * neurons_num
        sparsity = nonzeros / total_elements if total_elements > 0 else 1.0
        
        use_sparse = sparsity < 0.2  # Soglia automatica
        
        if use_sparse:
            # Crea tensori PyTorch sparse direttamente
            indices_stm = torch.tensor([stm_rows, stm_cols], dtype=torch.int64)
            values_stm = torch.tensor(stm_data, dtype=torch.int32)
            spikingTransitionMatrix = torch.sparse_coo_tensor(
                indices_stm, values_stm, (rule_num, neurons_num)
            ).coalesce()
            
            indices_sm = torch.tensor([sm_rows, sm_cols], dtype=torch.int64)
            values_sm = torch.tensor(sm_data, dtype=torch.int32)
            synapsesMatrix = torch.sparse_coo_tensor(
                indices_sm, values_sm, (rule_num, neurons_num)
            ).coalesce()
            
            print(f"Using PyTorch sparse matrices: {nonzeros} nonzeros ({sparsity:.2%} density)")
        else:
            # Crea array NumPy densi
            spikingTransitionMatrix = np.zeros((rule_num, neurons_num), dtype=np.int32)
            synapsesMatrix = np.zeros((rule_num, neurons_num), dtype=np.int32)
            
            for i in range(nonzeros):
                spikingTransitionMatrix[stm_rows[i], stm_cols[i]] = stm_data[i]
                synapsesMatrix[sm_rows[i], sm_cols[i]] = sm_data[i]
        
        # Input/Output neurons
        input_neurons_array = np.array(input_neurons, dtype=np.int32) if input_neurons else None
        output_neurons_array = np.array(output_neurons, dtype=np.int32) if output_neurons else None
        
        # Spike train
        single_spike_train = None
        if Config.MODE != "CNN" and snp_system.spike_train is not None:
            single_spike_train = np.asarray(snp_system.spike_train, dtype=np.int32)
        
        # Crea MSNPSystemExactGPU
        return MSNPSystemExactGPU(
            configurationVector=configurationVector,
            spikingVector=spikingVector,
            spikingTransitionMatrix=spikingTransitionMatrix,
            synapsesMatrix=synapsesMatrix,
            ruleVector=ruleVector,
            max_steps=snp_system.max_steps,
            deterministic=snp_system.deterministic,
            single_spike_train=single_spike_train,
            input_neurons=input_neurons_array,
            output_neurons=output_neurons_array,
            applyingRuleVector=applyingRuleVector,
            device=device,
            testsize=snp_system.input_len
        )