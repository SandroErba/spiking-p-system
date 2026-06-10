import numpy as np
import torch
from sps.snp_system import SNPSystem
from sps.config import Config
from sps.m_snp_pytorch_exact_GPU_and_CPU import MSNPSystemExactGPU


class MatrixExecutor:
    
    @staticmethod
    def translate_to_matrix(snp_system, device="cpu", use_sparse=False):
        """
        Traduzione ottimizzata con supporto PyTorch sparse nativo.
        
        Args:
            snp_system: SNPSystem instance
            device: "cpu" o "gpu"/"cuda"
            use_sparse: Se True, crea matrici ottimizzate per conversione sparse
        """
        neurons = snp_system.neurons
        neurons_num = len(neurons)
        
        # Pre-calcolo efficiente
        rule_counts = [len(neuron.transf_rules) for neuron in neurons]
        rule_num = sum(rule_counts)
        
        # Determina se usare sparse basato su dimensione e sparsità
        total_elements = rule_num * neurons_num
        
        # Vettori (sempre densi)
        configurationVector = np.zeros(neurons_num, dtype=np.int32)
        spikingVector = np.zeros(rule_num, dtype=np.int32)
        ruleVector = np.zeros(rule_num, dtype=np.int32)
        applyingRuleVector = np.zeros(rule_num, dtype=np.int32)
        
        # Per costruzione sparse, usiamo COO format direttamente
        stm_rows, stm_cols, stm_data = [], [], []
        sm_rows, sm_cols, sm_data = [], [], []
        
        input_neurons = []
        output_neurons = []
        rule_idx = 0
        non_zeros_count = 0
        
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
                
                non_zeros_count += 1
                
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
                        
                        non_zeros_count += 1
                
                ruleVector[rule_idx] = rule.mod
                applyingRuleVector[rule_idx] = nid
                rule_idx += 1
        
        # Decisione automatica su sparse vs dense
        sparsity = non_zeros_count / total_elements if total_elements > 0 else 1.0
        
        if use_sparse or (total_elements > 10000 and sparsity < 0.3):
            # Crea matrici in formato ottimizzato per PyTorch sparse
            stm_data_array = np.array(stm_data, dtype=np.int32)
            sm_data_array = np.array(sm_data, dtype=np.int32)
            
            # Crea array strutturati per conversione efficiente
            stm_indices = np.array([stm_rows, stm_cols], dtype=np.int64)
            sm_indices = np.array([sm_rows, sm_cols], dtype=np.int64)
            
            # Crea direttamente tensori PyTorch sparse
            spikingTransitionMatrix = torch.sparse_coo_tensor(
                torch.from_numpy(stm_indices),
                torch.from_numpy(stm_data_array.astype(np.float32 if 'cuda' in device else np.int32)),
                (rule_num, neurons_num)
            )
            
            synapsesMatrix = torch.sparse_coo_tensor(
                torch.from_numpy(sm_indices),
                torch.from_numpy(sm_data_array.astype(np.float32 if 'cuda' in device else np.int32)),
                (rule_num, neurons_num)
            )
            
            actual_use_sparse = True
            print(f"Created PyTorch sparse matrices: {non_zeros_count} nonzeros "
                  f"({sparsity:.2%} density)")
        else:
            # Matrici dense NumPy standard
            spikingTransitionMatrix = np.zeros((rule_num, neurons_num), dtype=np.int32)
            synapsesMatrix = np.zeros((rule_num, neurons_num), dtype=np.int32)
            
            for i in range(len(stm_rows)):
                spikingTransitionMatrix[stm_rows[i], stm_cols[i]] = stm_data[i]
                synapsesMatrix[sm_rows[i], sm_cols[i]] = sm_data[i]
            
            actual_use_sparse = False
        
        # Converte liste in array
        input_neurons_array = np.array(input_neurons, dtype=np.int32) if input_neurons else None
        output_neurons_array = np.array(output_neurons, dtype=np.int32) if output_neurons else None
        
        # Spike train
        single_spike_train = None
        if Config.MODE != "CNN" and snp_system.spike_train is not None:
            single_spike_train = np.asarray(snp_system.spike_train, dtype=np.int32)
        
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
            testsize=snp_system.input_len,
            use_sparse=actual_use_sparse
        )