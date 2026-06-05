import time
import os
import numpy as np
import torch

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import cnn
from sps.config import Config, database
from sps.m_matrix_executor_pytorch import MatrixExecutor

print("="*60)
print("CONFRONTO CPU vs GPU - MSNPSystemGPU")
print("="*60)

# Configurazione comune
database("digit")
Config.MODE = "CNN"
Config.compute_k_range()
Config.WHITE_HOLE = True 

# Carica il modello una volta sola
print("\nCaricamento modello SNPS...")
snps = cnn.test_launch_mnist_cnn()
print(f"Modello caricato: {len(snps.neurons)} neuroni")

# ============================================
# TEST SU CPU
# ============================================
print("\n" + "="*60)
print("TEST SU CPU")
print("="*60)

# Crea sistema (di default potrebbe usare GPU)
msnp_cpu = MatrixExecutor.translate_to_matrix(snps)

# Forza spostamento su CPU di TUTTI i tensori
msnp_cpu.to('cpu')
msnp_cpu.loadImages(snps.spike_train)

# Esecuzione CPU
start_cpu = time.perf_counter()
msnp_cpu.step(verbose=False)
end_cpu = time.perf_counter()
cpu_time = (end_cpu - start_cpu) * 1000

print(f"CPU completato in {cpu_time:.2f} ms")

# ============================================
# TEST SU GPU
# ============================================
print("\n" + "="*60)
print("TEST SU GPU")
print("="*60)

if torch.cuda.is_available():
    # Crea sistema
    msnp_gpu = MatrixExecutor.translate_to_matrix(snps)
    
    # Forza spostamento su GPU
    msnp_gpu.to('cuda')
    msnp_gpu.loadImages(snps.spike_train)
    
    # Sincronizza GPU prima dell'esecuzione
    torch.cuda.synchronize()
    
    # Esecuzione GPU
    start_gpu = time.perf_counter()
    msnp_gpu.step(verbose=False)
    torch.cuda.synchronize()
    end_gpu = time.perf_counter()
    gpu_time = (end_gpu - start_gpu) * 1000
    
    print(f"GPU completato in {gpu_time:.2f} ms")
else:
    print("GPU non disponibile")
    gpu_time = None

# ============================================
# CONFRONTO FINALE
# ============================================
print("\n" + "="*60)
print("CONFRONTO PERFORMANCE")
print("="*60)

print(f"\n{'Dispositivo':<15} {'Tempo (ms)':<15} {'Velocita':<15}")
print("-" * 45)
print(f"{'CPU':<15} {cpu_time:<15.2f} {'baseline':<15}")

if gpu_time:
    print(f"{'GPU':<15} {gpu_time:<15.2f} {'':<15}")
    speedup = cpu_time / gpu_time
    if speedup > 1:
        print(f"\nGPU e {speedup:.2f}x piu veloce della CPU")
        print(f"   Tempo risparmiato: {(1 - 1/speedup) * 100:.1f}%")
    else:
        print(f"\nCPU e {1/speedup:.2f}x piu veloce della GPU")
        print(f"   (problema troppo piccolo per GPU)")

print("\n" + "="*60)