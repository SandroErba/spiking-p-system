import os
import sys

# Forza la variabile d'ambiente
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'

# Aggiungi anche al PATH di sistema se necessario
os.environ['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;' + os.environ.get('PATH', '')

import numpy as np
import cupy as cp
import time
from sps.m_gpu import MSNPSystemGPU
from sps.config import Config
from sps.snp_system import SNPSystem
from sps.m_snp_system import MSNPSystem

print(f"GPU disponibile: {cp.cuda.is_available()}")
print(f"GPU attiva: {cp.cuda.runtime.getDevice()}")
print(f"Nome GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")

print("DEBUG: Import completati")

Config.WHITE_HOLE = False 

# Parametri
N_NEURONS = 1000
MAX_STEPS = 1000
print(f"DEBUG: N_NEURONS={N_NEURONS}, MAX_STEPS={MAX_STEPS}")

# Inizializzazione delle cariche dei neuroni
np.random.seed(42)
configurationVector = np.random.randint(1, 10, size=N_NEURONS, dtype=np.int32)
print(f"DEBUG: configurationVector creato, shape={configurationVector.shape}")

# Matrice di transizione spiking (N_NEURONS x N_NEURONS)
print("DEBUG: Creazione spikingTransitionMatrix...")
spikingTransitionMatrix = np.zeros((N_NEURONS, N_NEURONS), dtype=np.int32)
for i in range(N_NEURONS):
    for j in range(N_NEURONS):
        if i == j:
            spikingTransitionMatrix[i, j] = -2
        else:
            if np.random.random() < 0.3:
                spikingTransitionMatrix[i, j] = np.random.randint(1, 4)
            else:
                spikingTransitionMatrix[i, j] = 0
print(f"DEBUG: spikingTransitionMatrix creata, shape={spikingTransitionMatrix.shape}")

# Matrice delle sinapsi
print("DEBUG: Creazione synapsesMatrix...")
synapsesMatrix = np.zeros((N_NEURONS, N_NEURONS), dtype=np.int32)
for i in range(N_NEURONS):
    for j in range(N_NEURONS):
        if np.random.random() < 0.5:
            synapsesMatrix[i, j] = 1
print(f"DEBUG: synapsesMatrix creata, shape={synapsesMatrix.shape}")

# Matrice delle regole (N_NEURONS x 2) - chiamata ruleVector per compatibilita
print("DEBUG: Creazione ruleVectorCPU...")
ruleVectorCPU = np.zeros((N_NEURONS, 2), dtype=np.int32)
for i in range(N_NEURONS):
    ruleVectorCPU[i, 0] = 0
    ruleVectorCPU[i, 1] = np.random.randint(1, 5)
print(f"DEBUG: ruleVectorCPU creata, shape={ruleVectorCPU.shape}")

print("DEBUG: Creazione ruleVectorGPU...")
ruleVectorGPU = np.random.randint(1, 5, size=N_NEURONS, dtype=np.int32)
print(f"DEBUG: ruleVectorGPU creato, shape={ruleVectorGPU.shape}")

# Vettore delle regole da applicare
applyingRuleVector = np.arange(N_NEURONS, dtype=np.int32)
print(f"DEBUG: applyingRuleVector creato, shape={applyingRuleVector.shape}")

spikingVector = None
print(f"DEBUG: spikingVector={spikingVector}")

input_neurons = np.array([], dtype=np.int32)
single_spike_train = np.array([], dtype=np.int32)
print(f"DEBUG: input_neurons shape={input_neurons.shape}, single_spike_train shape={single_spike_train.shape}")

print(f"\nConfigurazione sistema con {N_NEURONS} neuroni")
print(f"   - Matrice transizione: {spikingTransitionMatrix.shape}")
print(f"   - Matrice sinapsi: {synapsesMatrix.shape}")
print(f"   - Matrice regole (ruleVectorGPU): {ruleVectorGPU.shape}")

# Create GPU system
print("\nDEBUG: Creazione MSNPSystemGPU...")
try:
    gpu_system = MSNPSystemGPU(
        configurationVector=configurationVector,
        spikingVector=spikingVector,
        spikingTransitionMatrix=spikingTransitionMatrix,
        synapsesMatrix=synapsesMatrix,
        ruleVector=ruleVectorGPU,
        max_steps=MAX_STEPS,
        deterministic=True,
        single_spike_train=single_spike_train,
        input_neurons=input_neurons,
        applyingRuleVector=applyingRuleVector
    )
    print("DEBUG: MSNPSystemGPU creato con successo")
except Exception as e:
    print(f"DEBUG: ERRORE nella creazione di MSNPSystemGPU: {e}")
    raise

print("DEBUG: Avvio esecuzione su GPU...")
start_time = time.time()
try:
    gpu_system.execute(verbose=False)
    cp.cuda.Stream.null.synchronize()
    print("DEBUG: Esecuzione GPU completata")
except Exception as e:
    print(f"DEBUG: ERRORE durante esecuzione GPU: {e}")
    raise
end_time = time.time()
gpu_time = end_time - start_time
print(f"GPU Execution time: {gpu_time:.4f} seconds")

# Preparazione per CPU - MSNPSystem richiede ruleMatrix
print("\nDEBUG: Creazione MSNPSystem (CPU)...")
try:
    msnp = MSNPSystem(
        configurationVector=configurationVector,
        spikingVector=spikingVector,
        spikingTransitionMatrix=spikingTransitionMatrix,
        ruleVector=ruleVectorCPU,
        max_steps=MAX_STEPS,
        deterministic=True,
        single_spike_train=single_spike_train,
        input_neurons=input_neurons,
        applyingRuleVector=applyingRuleVector,
        targetVector=np.zeros(N_NEURONS, dtype=np.int32),
        netGainVector=np.zeros_like(configurationVector)
    )
    print("DEBUG: MSNPSystem CPU creato con successo")
except Exception as e:
    print(f"DEBUG: ERRORE nella creazione di MSNPSystem CPU: {e}")
    raise

print("DEBUG: Avvio esecuzione su CPU...")
start_time = time.time()
try:
    msnp.execute(verbose=False)
    print("DEBUG: Esecuzione CPU completata")
except Exception as e:
    print(f"DEBUG: ERRORE durante esecuzione CPU: {e}")
    raise
end_time = time.time()
cpu_time = end_time - start_time
print(f"CPU Execution time: {cpu_time:.4f} seconds")

print("\nCONFRONTO PERFORMANCE:")
print(f"   CPU time: {cpu_time:.4f} secondi")
print(f"   GPU time: {gpu_time:.4f} secondi")
print(f"   Speedup GPU/CPU: {cpu_time/gpu_time:.2f}x")
print(f"   Accelerazione: {(1 - gpu_time/cpu_time)*100:.1f}% piu veloce su GPU")