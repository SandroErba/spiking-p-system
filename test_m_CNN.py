import time
import os
import numpy as np
import torch

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import cnn
from sps.config import Config, database
from sps.m_matrix_executor_pytorch import MatrixExecutor

print("="*60)
print("CONFRONTO CPU vs GPU - MULTIPLE ESECUZIONI")
print("="*60)

# Configurazione
database("digit")
Config.MODE = "CNN"
Config.compute_k_range()
Config.WHITE_HOLE = True 

# Carica modello
print("\nCaricamento modello...")
snps = cnn.test_launch_mnist_cnn()
print(f"Modello caricato: {len(snps.neurons)} neuroni")

# Numero di esecuzioni per la media
N_ITERATIONS = 5

# ============================================
# TEST CPU
# ============================================
print(f"\nTEST SU CPU ({N_ITERATIONS} esecuzioni)")
print("-" * 40)

cpu_times = []
for i in range(N_ITERATIONS):
    msnp_cpu = MatrixExecutor.translate_to_matrix(snps)
    msnp_cpu.device = torch.device('cpu')
    msnp_cpu.loadImages(snps.spike_train)
    
    start = time.perf_counter()
    msnp_cpu.step(verbose=False)
    end = time.perf_counter()
    
    cpu_time = (end - start) * 1000
    cpu_times.append(cpu_time)
    print(f"  Esecuzione {i+1}: {cpu_time:.2f} ms")

cpu_avg = np.mean(cpu_times)
cpu_std = np.std(cpu_times)
print(f"\nCPU: media = {cpu_avg:.2f} +- {cpu_std:.2f} ms")

# ============================================
# TEST GPU
# ============================================
if torch.cuda.is_available():
    print(f"\nTEST SU GPU ({N_ITERATIONS} esecuzioni)")
    print("-" * 40)
    
    gpu_times = []
    for i in range(N_ITERATIONS):
        msnp_gpu = MatrixExecutor.translate_to_matrix(snps)
        msnp_gpu.device = torch.device('cuda')
        msnp_gpu.loadImages(snps.spike_train)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        msnp_gpu.step(verbose=False)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        gpu_time = (end - start) * 1000
        gpu_times.append(gpu_time)
        print(f"  Esecuzione {i+1}: {gpu_time:.2f} ms")
    
    gpu_avg = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)
    print(f"\nGPU: media = {gpu_avg:.2f} +- {gpu_std:.2f} ms")
    
    # ============================================
    # CONFRONTO
    # ============================================
    print("\n" + "="*60)
    print("CONFRONTO FINALE")
    print("="*60)
    
    print(f"\n{'Metrica':<20} {'CPU':>15} {'GPU':>15}")
    print("-" * 50)
    print(f"{'Tempo medio (ms)':<20} {cpu_avg:>15.2f} {gpu_avg:>15.2f}")
    print(f"{'Deviazione std (ms)':<20} {cpu_std:>15.2f} {gpu_std:>15.2f}")
    
    speedup = cpu_avg / gpu_avg
    print(f"\n{'Speedup':<20} {1:>15.1f}x {speedup:>15.2f}x")
    
    if speedup > 1:
        print(f"\nGPU e {speedup:.2f}x piu veloce della CPU")
        print(f"   Tempo risparmiato: {(1 - 1/speedup) * 100:.1f}%")
    else:
        print(f"\nCPU e {1/speedup:.2f}x piu veloce della GPU")
        print(f"   (problema troppo piccolo per GPU)")
else:
    print("\nGPU non disponibile")

print("\n" + "="*60)