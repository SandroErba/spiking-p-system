import time
import os
import sys
import traceback
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
import csv
from pathlib import Path


class AccuracyLogger:
    """Salva le accuracy in un CSV."""
    
    DIR_NAME = "results"
    
    def __init__(self, filename="accuracy_results.csv"):
        self.filename = filename
        self.rows = []
        self.base_dir = Path.cwd()
        
    def add_result(self, system, device, q_range, test_num, seed, train_size, test_size, accuracy, error=None):
        """Aggiunge un risultato."""
        self.rows.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'system': system,
            'device': device,
            'Q_RANGE': q_range,
            'TEST_NUM': test_num,
            'SEED': seed,
            'TRAIN_SIZE': train_size,
            'TEST_SIZE': test_size,
            'ACCURACY': f"{accuracy:.4f}" if error is None else 'ERROR',
            'ERROR': str(error)[:200] if error else ''
        })
    
    def save(self):
        """Salva tutti i risultati in CSV."""
        if not self.rows:
            print("No results to save")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.base_dir / self.DIR_NAME
        results_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = results_dir / f"{self.filename.replace('.csv', '')}_{timestamp}.csv"
        
        fieldnames = ['timestamp', 'system', 'device', 'Q_RANGE', 'TEST_NUM', 'SEED', 
                      'TRAIN_SIZE', 'TEST_SIZE', 'ACCURACY', 'ERROR']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        
        print(f"Accuracy results saved to: {csv_path}")
        print(f"Total results: {len(self.rows)}")
        return csv_path


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


def emergency_save(generalTimer, accuracyLogger, error_msg=""):
    """Salvataggio di emergenza dei risultati."""
    try:
        # Salva timer
        generalTimer.export_to_csv(False)
        
        # Salva accuracy
        accuracyLogger.save()
        
        # Salva log errore
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = f"error_log_{timestamp}.txt"
        with open(error_file, 'w') as f:
            f.write(f"Error: {error_msg}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
        
        print(f"\n{'!'*50}")
        print(f"EMERGENCY SAVE completato!")
        print(f"Timer e Accuracy salvati")
        print(f"Error log: {error_file}")
        print(f"{'!'*50}")
        
    except Exception as e:
        print(f"EMERGENCY SAVE FAILED: {e}")


# ============================================================
# MAIN
# ============================================================

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
    accuracyLogger = AccuracyLogger("accuracy_results.csv")
    
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
                    
                    generalTimer.start_step(f"Q:{Q_RANGE}_T:{TEST_NUM}_S{size[0]}_SNPS_CPU")
                    accuracy = network.launch_mnist("SNPSystem", "cpu")
                    generalTimer.end_step()
                    
                    # Salva risultato
                    accuracyLogger.add_result("SNPS", "CPU", Q_RANGE, TEST_NUM, SEED, 
                                             size[0], size[1], accuracy)
                    print(f"<<< DONE [SNPS - CPU] Accuracy: {accuracy:.4f}")
                    
                except Exception as e:
                    error_msg = f"SNPS CPU - Q:{Q_RANGE} T:{TEST_NUM} Size:{size}"
                    print(f"\n{'!'*50}")
                    print(f"ERROR: {error_msg}")
                    print(traceback.format_exc())
                    print(f"{'!'*50}")
                    accuracyLogger.add_result("SNPS", "CPU", Q_RANGE, TEST_NUM, SEED,
                                             size[0], size[1], None, error=error_msg)
                    emergency_save(generalTimer, accuracyLogger, error_msg)
                
                # === MSNPS GPU ===
                try:
                    reset_gpu_for_rerun()
                    kill_all_cuda_contexts()
                    
                    print(f"\n>>> RUNNING [MSNPS - GPU] Q:{Q_RANGE} T:{TEST_NUM}")
                    generalTimer.start_step(f"Q:{Q_RANGE}_T:{TEST_NUM}_S{size[0]}_MSNPS_GPU")
                    accuracy = network.launch_mnist("MSNPSystemExactGPU", "gpu")
                    generalTimer.end_step()
                    
                    accuracyLogger.add_result("MSNPS", "GPU", Q_RANGE, TEST_NUM, SEED,
                                             size[0], size[1], accuracy)
                    print(f"<<< DONE [MSNPS - GPU] Accuracy: {accuracy:.4f}")
                    
                except Exception as e:
                    error_msg = f"MSNPS GPU - Q:{Q_RANGE} T:{TEST_NUM} Size:{size}"
                    print(f"\n{'!'*50}")
                    print(f"ERROR: {error_msg}")
                    print(traceback.format_exc())
                    print(f"{'!'*50}")
                    accuracyLogger.add_result("MSNPS", "GPU", Q_RANGE, TEST_NUM, SEED,
                                             size[0], size[1], None, error=error_msg)
                    emergency_save(generalTimer, accuracyLogger, error_msg)
                
                # === MSNPS CPU ===
                try:
                    reset_gpu_for_rerun()
                    kill_all_cuda_contexts()
                    
                    print(f"\n>>> RUNNING [MSNPS - CPU] Q:{Q_RANGE} T:{TEST_NUM}")
                    generalTimer.start_step(f"Q:{Q_RANGE}_T:{TEST_NUM}_S{size[0]}_MSNPS_CPU")
                    accuracy = network.launch_mnist("MSNPSystemExactGPU", "cpu")
                    generalTimer.end_step()
                    
                    accuracyLogger.add_result("MSNPS", "CPU", Q_RANGE, TEST_NUM, SEED,
                                             size[0], size[1], accuracy)
                    print(f"<<< DONE [MSNPS - CPU] Accuracy: {accuracy:.4f}")
                    
                except Exception as e:
                    error_msg = f"MSNPS CPU - Q:{Q_RANGE} T:{TEST_NUM} Size:{size}"
                    print(f"\n{'!'*50}")
                    print(f"ERROR: {error_msg}")
                    print(traceback.format_exc())
                    print(f"{'!'*50}")
                    accuracyLogger.add_result("MSNPS", "CPU", Q_RANGE, TEST_NUM, SEED,
                                             size[0], size[1], None, error=error_msg)
                    emergency_save(generalTimer, accuracyLogger, error_msg)
                
                # Salva risultati parziali ogni 9 test (3 sistemi x 3 size)
                test_counter = (Q_RANGE - 2) * TEST_PER_Q * len(sizes) + TEST_NUM * len(sizes)
                for size_idx in range(len(sizes)):
                    if (test_counter + size_idx + 1) % 9 == 0:
                        accuracyLogger.save()
                        print("Partial results saved!")

except KeyboardInterrupt:
    print("\n\nINTERRUPTED BY USER!")
    error_msg = "KeyboardInterrupt"
    emergency_save(generalTimer, accuracyLogger, error_msg)
    
except Exception as e:
    print("\n\nFATAL ERROR!")
    error_msg = f"FATAL: {str(e)}"
    print(traceback.format_exc())
    emergency_save(generalTimer, accuracyLogger, error_msg)

finally:
    # Salva sempre alla fine
    print("\n\n" + "="*50)
    print("SAVING FINAL RESULTS...")
    try:
        generalTimer.export_to_csv(False)
        accuracyLogger.save()
        print("Final results saved successfully!")
    except Exception as e:
        print(f"Error saving final results: {e}")
    
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
