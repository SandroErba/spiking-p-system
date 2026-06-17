import time
import os
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datetime import datetime

from sps import  other_networks, network, flower_image, digit_image, med_image, handle_csv
from sps.config import Config, database
#from sps.m_matrix_executor import MatrixExecutor
#from sps.m_snp_system import MSNPSystem
#from sps.snp_system import SNPSystem
import torch
import gc
from sps.timersnp import TimerSNP
import random


def reset_gpu_for_rerun():
    """Prepara la GPU per una nuova esecuzione azzerando la memoria."""
    
    # 1. Sincronizza GPU (completa tutte le operazioni pendenti)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 2. Elimina tutti i tensori CUDA ancora in memoria
    # Questo cattura anche tensori in liste, dict, etc.
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    # Sposta su CPU prima di eliminare (rompe riferimenti circolari)
                    obj.data = obj.data.cpu()
                    del obj
        except:
            pass
    
    # 3. Pulisci i moduli PyTorch che potrebbero avere buffer CUDA
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith('sps.'):
            for attr_name in dir(mod):
                try:
                    attr = getattr(mod, attr_name)
                    if hasattr(attr, '_buffers'):
                        for buf in attr._buffers.values():
                            if torch.is_tensor(buf) and buf.is_cuda:
                                buf.data = buf.data.cpu()
                except:
                    pass
    
    # 4. Garbage collection aggressiva
    gc.collect()
    gc.collect()  # Doppia passata per oggetti con __del__
    
    # 5. Svuota cache CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Pulisci memoria condivisa tra processi
    
    # 6. Sincronizza di nuovo
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 7. Stampa stato memoria GPU (debug)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory after reset - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def kill_all_cuda_contexts():
    """Metodo nucleare: ricrea il contesto CUDA da zero."""
    if torch.cuda.is_available():
        # Salva il device count
        device_count = torch.cuda.device_count()
        
        # Svuota tutto
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        # Forza la deallocazione di tutti i tensori
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # Ricrea il contesto (questo è il metodo più aggressivo)
        for i in range(device_count):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        print("CUDA contexts reset completely")

try:
    # Inizializzazione
    reset_gpu_for_rerun()
    
    database("digit")
    Config.compute_k_range()
    
    sizes = [[50,50],[1000,100],[5000,2500]]
    
    Q_LIMIT = 8
    TEST_PER_Q = 3
    SEEDS = [42, 999, 1234]
    
    generalTimer = TimerSNP(Q_LIMIT * TEST_PER_Q * len(sizes) * 30, "ElapsedTimePerSystemAndQrange", True)
    
    # Loop principale
    for Q_RANGE in range(2, Q_LIMIT + 1):
        for TEST_NUM in range(0, TEST_PER_Q):
            
            # Set seed
            SEED = SEEDS[TEST_NUM]
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            
            for size in sizes:
                Config.TRAIN_SIZE = size[0]
                Config.TEST_SIZE = size[1]
                
                print("\n" + "#"*50)
                print(f"TRAIN SIZE: {Config.TRAIN_SIZE} | TEST SIZE: {Config.TEST_SIZE}")
                print(f"Q: {Q_RANGE} | TEST: {TEST_NUM} | SEED: {SEED}")
                print("#"*50)
                
                # === SNPS CPU ===
                try:
                    reset_gpu_for_rerun()
                    kill_all_cuda_contexts()
                    
                    print(f"\n>>> RUNNING [SNPS - CPU] Q:{Q_RANGE} T:{TEST_NUM}")
                    Config.TIME_TEST_NUM = TEST_NUM
                    Config.Q_RANGE = Q_RANGE
                    
                    generalTimer.start_step(f"Q:{Q_RANGE}_T:{TEST_NUM}_Size:{size[0]}_SNPS_CPU")
                    network.launch_mnist("SNPSystem", "cpu")
                    generalTimer.end_step()
                    print(f"<<< DONE [SNPS - CPU] Q:{Q_RANGE} T:{TEST_NUM}")
                    
                except Exception as e:
                    error_msg = f"SNPS CPU - Q:{Q_RANGE} T:{TEST_NUM} Size:{size} - {str(e)}"
                    print(f"\n{'!'*50}")
                    print(f"ERROR: {error_msg}")
                    print(traceback.format_exc())
                    print(f"{'!'*50}")
                    emergency_save(generalTimer, error_msg)
                
                # === MSNPS GPU ===
                try:
                    reset_gpu_for_rerun()
                    kill_all_cuda_contexts()
                    
                    print(f"\n>>> RUNNING [MSNPS - GPU] Q:{Q_RANGE} T:{TEST_NUM}")
                    generalTimer.start_step(f"Q:{Q_RANGE}_T:{TEST_NUM}_Size:{size[0]}_MSNPS_GPU")
                    network.launch_mnist("MSNPSystemExactGPU", "gpu")
                    generalTimer.end_step()
                    print(f"<<< DONE [MSNPS - GPU] Q:{Q_RANGE} T:{TEST_NUM}")
                    
                except Exception as e:
                    error_msg = f"MSNPS GPU - Q:{Q_RANGE} T:{TEST_NUM} Size:{size} - {str(e)}"
                    print(f"\n{'!'*50}")
                    print(f"ERROR: {error_msg}")
                    print(traceback.format_exc())
                    print(f"{'!'*50}")
                    emergency_save(generalTimer, error_msg)
                
                # === MSNPS CPU ===
                try:
                    reset_gpu_for_rerun()
                    kill_all_cuda_contexts()
                    
                    print(f"\n>>> RUNNING [MSNPS - CPU] Q:{Q_RANGE} T:{TEST_NUM}")
                    generalTimer.start_step(f"Q:{Q_RANGE}_T:{TEST_NUM}_Size:{size[0]}_MSNPS_CPU")
                    network.launch_mnist("MSNPSystemExactGPU", "cpu")
                    generalTimer.end_step()
                    print(f"<<< DONE [MSNPS - CPU] Q:{Q_RANGE} T:{TEST_NUM}")
                    
                except Exception as e:
                    error_msg = f"MSNPS CPU - Q:{Q_RANGE} T:{TEST_NUM} Size:{size} - {str(e)}"
                    print(f"\n{'!'*50}")
                    print(f"ERROR: {error_msg}")
                    print(traceback.format_exc())
                    print(f"{'!'*50}")
                    emergency_save(generalTimer, error_msg)

except KeyboardInterrupt:
    print("\n\nINTERRUPTED BY USER!")
    error_msg = "KeyboardInterrupt"
    emergency_save(generalTimer, error_msg)
    
except Exception as e:
    print("\n\nFATAL ERROR!")
    error_msg = f"FATAL: {str(e)}"
    print(traceback.format_exc())
    emergency_save(generalTimer, error_msg)

finally:
    # Salva sempre alla fine
    print("\n\n" + "="*50)
    print("SAVING FINAL RESULTS...")
    try:
        generalTimer.export_to_csv(False)
        print("Final results saved successfully!")
    except Exception as e:
        print(f"Error saving final results: {e}")
        emergency_save(generalTimer, str(e))
    
    print("="*50)
    print("EXECUTION COMPLETED")

# ====== ARCHIVIO =========

# print("="*30)
# print("SNP System - CPU")
# #snps = network.create_exact_csv()
# t = time.perf_counter()
# network.launch_mnist("SNPSystem", "cpu")
# print("---> Elapsed:", time.perf_counter() - t)

# print("="*30)
# print("MSNP System - GPU")
# t = time.perf_counter()
# network.launch_mnist("MSNPSystemExactGPU", "gpu")
# print("---> Elapsed 2:", time.perf_counter() - t)
# print("="*30)
# print("MSNP System - CPU")

# t = time.perf_counter()
# network.launch_mnist("MSNPSystemExactGPU", "cpu")
# print("---> Elapsed 3:", time.perf_counter() - t)


#MSNPSystemDivModGPU, MSNPSystemExactGPU, SNPSystem

#network.launch_mnist_from_csv("SNPS_cnn_external.csv")

#Config.NUM_LAYERS = 6
#network.launch_mnist_from_csv("SNPS_deep_cnn.csv")

#other_networks.compute_extended() #require halting mode
#other_networks.compute_divisible_3() #require halting mode
#other_networks.compute_gen_even() #require generative mode
#other_networks.prova() #require halting mode
