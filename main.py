import time
import os
import sys
import traceback
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datetime import datetime

from sps import  other_networks, network, flower_image, digit_image, med_image, handle_csv
from sps.config import Config, database
import torch
import gc
from sps.timersnp import TimerSNP
import random
import csv
from pathlib import Path


class AccuracyLogger:
    """Salva le accuracy in un CSV."""
    
    DIR_NAME = "results"
    
    def __init__(self, filename="accuracy_results.csv", overwrite=True):
        self.filename = filename
        self.rows = []
        self.base_dir = Path.cwd()
        self.overwrite = overwrite
        
        if self.overwrite:
            self._load_existing_results()
    
    def _load_existing_results(self):
        """Carica i risultati dal file più recente se esiste."""
        results_dir = self.base_dir / self.DIR_NAME
        
        if not results_dir.exists():
            return
        
        base_name = self.filename.replace('.csv', '')
        existing_files = list(results_dir.glob(f"{base_name}*.csv"))
        
        if existing_files:
            latest_file = max(existing_files, key=lambda x: x.stat().st_mtime)
            print(f"Loading existing results from: {latest_file}")
            
            try:
                with open(latest_file, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        self.rows.append(row)
                print(f"Loaded {len(self.rows)} existing results")
            except Exception as e:
                print(f"Error loading existing results: {e}")
                self.rows = []
    
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
        """Salva tutti i risultati in CSV, sovrascrivendo il file precedente se richiesto."""
        if not self.rows:
            print("No results to save")
            return None
        
        results_dir = self.base_dir / self.DIR_NAME
        results_dir.mkdir(parents=True, exist_ok=True)
        
        if self.overwrite:
            csv_path = results_dir / self.filename
            print(f"Overwriting existing file: {csv_path}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    
    def save_backup(self):
        """Salva una copia di backup con timestamp (opzionale)."""
        if not self.rows:
            return None
        
        results_dir = self.base_dir / self.DIR_NAME
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = results_dir / f"{self.filename.replace('.csv', '')}_backup_{timestamp}.csv"
        
        fieldnames = ['timestamp', 'system', 'device', 'Q_RANGE', 'TEST_NUM', 'SEED', 
                      'TRAIN_SIZE', 'TEST_SIZE', 'ACCURACY', 'ERROR']
        
        with open(backup_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        
        print(f"Backup saved to: {backup_path}")
        return backup_path


def reset_gpu_for_rerun():
    """Prepara la GPU per una nuova esecuzione azzerando la memoria."""
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    obj.data = obj.data.cpu()
                    del obj
        except:
            pass
    
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
    
    gc.collect()
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory after reset - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def kill_all_cuda_contexts():
    """Metodo nucleare: ricrea il contesto CUDA da zero."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        for i in range(device_count):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        print("CUDA contexts reset completely")


def emergency_save(generalTimer, accuracyLogger, error_msg=""):
    """Salvataggio di emergenza dei risultati."""
    try:
        generalTimer.export_to_csv(False)
        accuracyLogger.save()
        
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


def run_system_test(system_name, device, Q_RANGE, TEST_NUM, SEED, size, generalTimer, accuracyLogger):
    """Esegue un singolo test per un sistema specifico."""
    Config.TRAIN_SIZE = size[0]
    Config.TEST_SIZE = size[1]
    
    print("\n" + "#"*50)
    print(f"SYSTEM: {system_name} | DEVICE: {device}")
    print(f"TRAIN SIZE: {Config.TRAIN_SIZE} | TEST SIZE: {Config.TEST_SIZE}")
    print(f"Q: {Q_RANGE} | TEST: {TEST_NUM} | SEED: {SEED}")
    print("#"*50)
    
    system_label = "SNPS" if "SNPSystem" in system_name else "MSNPS"
    
    try:
        reset_gpu_for_rerun()
        kill_all_cuda_contexts()
        
        # Set seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"\n>>> RUNNING [{system_label} - {device.upper()}] Q:{Q_RANGE} T:{TEST_NUM}")
        Config.TIME_TEST_NUM = TEST_NUM
        Config.Q_RANGE = Q_RANGE
        
        step_name = f"Q:{Q_RANGE}_T:{TEST_NUM}_S{size[0]}_{system_label}_{device.upper()}"
        generalTimer.start_step(step_name)
        accuracy = network.launch_mnist(system_name, device)
        generalTimer.end_step()
        
        accuracyLogger.add_result(system_label, device.upper(), Q_RANGE, TEST_NUM, SEED, 
                                 size[0], size[1], accuracy)
        print(f"<<< DONE [{system_label} - {device.upper()}] Accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        error_msg = f"{system_label} {device.upper()} - Q:{Q_RANGE} T:{TEST_NUM} Size:{size}"
        print(f"\n{'!'*50}")
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())
        print(f"{'!'*50}")
        accuracyLogger.add_result(system_label, device.upper(), Q_RANGE, TEST_NUM, SEED,
                                 size[0], size[1], None, error=error_msg)
        emergency_save(generalTimer, accuracyLogger, error_msg)
        return False


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
    
    total_tests = Q_LIMIT * TEST_PER_Q * len(sizes) * 3  # 3 sistemi
    generalTimer = TimerSNP(total_tests, "ElapsedTimePerSystemAndQrange", True)
    accuracyLogger = AccuracyLogger("accuracy_results.csv", True)
    
    test_counter = 0
    
    # ========================================================
    # FASE 1: TUTTI GLI SNPS CPU
    # ========================================================
    print("\n" + "="*60)
    print("FASE 1: ESECUZIONE DI TUTTI GLI SNPS CPU")
    print("="*60)

    for size in sizes:
        for Q_RANGE in range(2, Q_LIMIT + 1):
            for TEST_NUM in range(0, TEST_PER_Q):
            
                reset_gpu_for_rerun()
                SEED = SEEDS[TEST_NUM]
                
                run_system_test("SNPSystem", "cpu", Q_RANGE, TEST_NUM, SEED, size, 
                               generalTimer, accuracyLogger)
                
                test_counter += 1
                if test_counter % 9 == 0:
                    accuracyLogger.save()
                    print(f"Partial results saved! ({test_counter} tests completed)")
    
    # ========================================================
    # FASE 2: TUTTI GLI MSNPS GPU
    # ========================================================
    print("\n" + "="*60)
    print("FASE 2: ESECUZIONE DI TUTTI GLI MSNPS GPU")
    print("="*60)
    
    for size in sizes:
        for Q_RANGE in range(2, Q_LIMIT + 1):
            for TEST_NUM in range(0, TEST_PER_Q):
                reset_gpu_for_rerun()
                kill_all_cuda_contexts()
                SEED = SEEDS[TEST_NUM]
                
                run_system_test("MSNPSystemExactGPU", "gpu", Q_RANGE, TEST_NUM, SEED, size, 
                               generalTimer, accuracyLogger)
                
                test_counter += 1
                if test_counter % 9 == 0:
                    accuracyLogger.save()
                    print(f"Partial results saved! ({test_counter} tests completed)")
    
    # ========================================================
    # FASE 3: TUTTI GLI MSNPS CPU
    # ========================================================
    print("\n" + "="*60)
    print("FASE 3: ESECUZIONE DI TUTTI GLI MSNPS CPU")
    print("="*60)
    
    for size in sizes:
        for Q_RANGE in range(2, Q_LIMIT + 1):
            for TEST_NUM in range(0, TEST_PER_Q):
                reset_gpu_for_rerun()
                kill_all_cuda_contexts()
                SEED = SEEDS[TEST_NUM]
                
                run_system_test("MSNPSystemExactGPU", "cpu", Q_RANGE, TEST_NUM, SEED, size, 
                               generalTimer, accuracyLogger)
                
                test_counter += 1
                if test_counter % 9 == 0:
                    accuracyLogger.save()
                    print(f"Partial results saved! ({test_counter} tests completed)")

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
        accuracyLogger.save_backup()  # Backup finale
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
