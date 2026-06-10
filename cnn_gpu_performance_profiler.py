"""
Diagnostica performance GPU per MSNP System
Analisi overhead e profilazione completa
"""

import torch
import time
import numpy as np
from torch.profiler import profile, ProfilerActivity, record_function
import os
import json

from sps.config import Config
from sps.exact_csv import SNPS_exact_csv
from sps.snp_system import SNPSystem
from sps.m_matrix_executor_exact import MatrixExecutor as MatrixExecutorExact


class MSNPGPUDiagnostic:
    """
    Classe per diagnosticare overhead GPU in MSNP System
    """
    
    def __init__(self, x_train):
        """
        Args:
            x_train: Dati di training (spike train)
        """
        self.x_train = x_train
        self.results = {}
        
        print("="*80)
        print("MSNP GPU PERFORMANCE DIAGNOSTIC")
        print("="*80)
        
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. Cannot run GPU diagnostic.")
            raise RuntimeError("CUDA not available")
        
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print("="*80)
    
    def diagnose_gpu_profiling(self, max_steps=None, output_dir="./profiler_output"):
        """
        Diagnostica GPU: Profiling completo con torch.profiler
        
        Args:
            max_steps: Numero massimo di step (None = usa default)
            output_dir: Directory per salvare i trace
        
        Returns:
            dict: Statistiche dettagliate
        """
        print("\n" + "="*80)
        print("GPU PROFILING DIAGNOSTIC")
        print("="*80)
        
        # Clean CUDA cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize system
        print("Initializing system...")
        SNPS_exact_csv()
        snps = SNPSystem()
        snps.load_neurons_from_csv("csv/" + Config.CSV_EXACT_NAME)
        
        # Translation to matrix on GPU
        start_translation = time.time()
        msnps = MatrixExecutorExact.translate_to_matrix(snps, device='gpu')
        translation_time = time.time() - start_translation
        print(f"Translation completed in {translation_time:.2f}s")
        
        # Load images
        msnps.loadImages(self.x_train)
        
        # Profiling
        print(f"Starting GPU profiling...")
        
        profile_stats = {}
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(output_dir, "gpu_trace")
            )
        ) as prof:
            
            with record_function("gpu_execution"):
                if max_steps:
                    original_max_steps = msnps.max_steps
                    msnps.max_steps = max_steps
                    result = msnps.execute(verbose=False)
                    msnps.max_steps = original_max_steps
                else:
                    result = msnps.execute(verbose=False)
                
                torch.cuda.synchronize()
        
        print("Profiling completed")
        
        # ===== ANALYSIS =====
        print("\n" + "-"*80)
        print("GPU PERFORMANCE ANALYSIS")
        print("-"*80)
        
        # 1. Slowest operations
        print("\nTOP 15 SLOWEST OPERATIONS:")
        table = prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=15
        )
        print(table)
        
        # 2. Critical overhead operations
        print("\nCRITICAL CPU-GPU OVERHEAD OPERATIONS:")
        suspicious_ops = ['to', 'copy', 'item', 'cpu', 'cuda', 'synchronize', 'memcpy', 'contiguous']
        critical_ops = []
        
        for op in prof.key_averages():
            op_name = str(op.key).lower()
            if any(s in op_name for s in suspicious_ops):
                total_time = op.self_cuda_time_total / 1000  # ms
                
                if total_time > 1.0:  # >1ms
                    critical_ops.append({
                        'name': op.key,
                        'calls': op.count,
                        'time_ms': total_time,
                        'cpu_time_ms': op.self_cpu_time_total / 1000,
                        'cuda_time_ms': op.self_cuda_time_total / 1000,
                        'shapes': op.input_shapes[:2] if op.input_shapes else []
                    })
        
        if critical_ops:
            for op in critical_ops[:10]:
                print(f"\n  - {op['name']}")
                print(f"    Calls: {op['calls']}")
                print(f"    Total time: {op['time_ms']:.2f}ms")
                print(f"    CUDA time: {op['cuda_time_ms']:.2f}ms")
                print(f"    CPU time: {op['cpu_time_ms']:.2f}ms")
                if op['shapes']:
                    print(f"    Typical shape: {op['shapes']}")
        else:
            print("  No critical operations detected (>1ms)")
        
        # 3. .item() calls analysis
        print("\n.ITEM() CALLS ANALYSIS:")
        item_ops = [op for op in prof.key_averages() if 'item' in str(op.key).lower()]
        if item_ops:
            total_item_calls = sum(op.count for op in item_ops)
            total_item_time = sum(op.self_cuda_time_total for op in item_ops) / 1000
            
            print(f"  CRITICAL: {total_item_calls} .item() calls detected!")
            print(f"     Total time spent: {total_item_time:.2f}ms")
            print(f"     WARNING: Each .item() forces CPU-GPU synchronization")
            
            for op in item_ops[:3]:
                print(f"     - {op.key}: {op.count} calls")
        else:
            print("  No .item() calls detected")
        
        # 4. Memory statistics
        print("\nGPU MEMORY STATISTICS:")
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Memory copy analysis
        memcpy_ops = [op for op in prof.key_averages() if 'memcpy' in str(op.key).lower()]
        if memcpy_ops:
            total_memcpy_time = sum(op.self_cuda_time_total for op in memcpy_ops) / 1000
            print(f"\n  MEMORY TRANSFERS:")
            print(f"     Total copy time: {total_memcpy_time:.2f}ms")
            for op in memcpy_ops[:3]:
                print(f"     - {op.key}: {op.count} copies, {op.self_cuda_time_total/1000:.2f}ms")
        
        # 5. GPU utilization
        print("\nGPU UTILIZATION ANALYSIS:")
        total_cuda_time = sum(op.self_cuda_time_total for op in prof.key_averages()) / 1000
        compute_ops = ['addmm', 'matmul', 'mm', 'bmm', 'convolution', 'add', 'mul', 'div']
        compute_time = sum(
            op.self_cuda_time_total for op in prof.key_averages() 
            if any(c in str(op.key).lower() for c in compute_ops)
        ) / 1000
        
        compute_percent = (compute_time / total_cuda_time * 100) if total_cuda_time > 0 else 0
        print(f"  Total CUDA time: {total_cuda_time:.2f}ms")
        print(f"  Actual computation time: {compute_time:.2f}ms ({compute_percent:.1f}%)")
        print(f"  Overhead/transfer time: {total_cuda_time - compute_time:.2f}ms ({100-compute_percent:.1f}%)")
        
        if compute_percent < 50:
            print(f"     WARNING: GPU is UNDERUTILIZED! More than half time is overhead")
        
        # 6. General statistics
        print("\nGENERAL STATISTICS:")
        total_cpu_time = sum(op.self_cpu_time_total for op in prof.key_averages()) / 1000
        print(f"  Total CPU time: {total_cpu_time:.2f}ms")
        print(f"  Total steps executed: {msnps.t_step}")
        if msnps.t_step > 0:
            print(f"  Average time per step: {total_cuda_time/msnps.t_step:.2f}ms")
        
        # Save statistics
        profile_stats = {
            'translation_time': translation_time,
            'total_cpu_time_ms': total_cpu_time,
            'total_cuda_time_ms': total_cuda_time,
            'compute_time_ms': compute_time,
            'compute_percent': compute_percent,
            'item_calls': total_item_calls if item_ops else 0,
            'total_steps': msnps.t_step,
            'critical_ops': critical_ops[:5]
        }
        
        self.results = profile_stats
        
        print("\n" + "="*80)
        print("GPU PROFILING COMPLETED")
        print(f"  Traces saved in: {output_dir}/gpu_trace")
        print("="*80)
        
        return profile_stats, prof
    
    def diagnose_gpu_only(self, num_runs=3, max_steps=None):
        """
        GPU-only diagnostic: misura performance GPU senza confronto CPU
        
        Args:
            num_runs: Numero di esecuzioni per una media stabile
            max_steps: Numero massimo di step (None = usa default)
        
        Returns:
            dict: Risultati GPU
        """
        print("\n" + "="*80)
        print("GPU PERFORMANCE DIAGNOSTIC")
        print("="*80)
        
        results = {
            'execution_times': [],
            'translation_times': [],
            'step_counts': [],
            'memory_usage': []
        }
        
        print(f"\nTesting GPU ({num_runs} runs)...")
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...", end=" ", flush=True)
            
            # Clean cache before each run
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            # Rebuild system
            SNPS_exact_csv()
            snps = SNPSystem()
            snps.load_neurons_from_csv("csv/" + Config.CSV_EXACT_NAME)
            
            # Measure translation
            start_trans = time.time()
            msnps = MatrixExecutorExact.translate_to_matrix(snps, device='gpu')
            trans_time = time.time() - start_trans
            results['translation_times'].append(trans_time)
            
            msnps.loadImages(self.x_train)
            
            # Modify max_steps if needed
            if max_steps:
                original_max = msnps.max_steps
                msnps.max_steps = max_steps
            
            # Execute with precise GPU timing
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
            msnps.execute(verbose=False)
            
            end_event.record()
            torch.cuda.synchronize()
            exec_time = start_event.elapsed_time(end_event) / 1000  # seconds
            
            # Restore
            if max_steps:
                msnps.max_steps = original_max
            
            results['execution_times'].append(exec_time)
            results['step_counts'].append(msnps.t_step)
            
            # Memory stats
            mem_allocated = torch.cuda.memory_allocated() / 1024**2
            mem_reserved = torch.cuda.memory_reserved() / 1024**2
            results['memory_usage'].append({
                'allocated_mb': mem_allocated,
                'reserved_mb': mem_reserved,
                'peak_mb': torch.cuda.max_memory_allocated() / 1024**2
            })
            
            print(f"Done: {exec_time:.3f}s, {msnps.t_step} steps, {mem_allocated:.1f}MB")
            
            # Clean up
            del msnps
            torch.cuda.empty_cache()
        
        # ===== ANALYSIS =====
        print("\n" + "-"*80)
        print("GPU PERFORMANCE REPORT")
        print("-"*80)
        
        # Execution time statistics
        avg_time = np.mean(results['execution_times'])
        std_time = np.std(results['execution_times'])
        avg_trans = np.mean(results['translation_times'])
        avg_steps = int(np.mean(results['step_counts']))
        
        print(f"\nGPU EXECUTION STATISTICS (over {num_runs} runs):")
        print(f"  Translation time: {avg_trans:.3f}s")
        print(f"  Execution time: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Steps executed: {avg_steps}")
        print(f"  Time per step: {avg_time/avg_steps*1000:.2f}ms" if avg_steps > 0 else "  Time per step: N/A")
        
        # Memory statistics
        avg_mem_alloc = np.mean([m['allocated_mb'] for m in results['memory_usage']])
        peak_mem = max([m['peak_mb'] for m in results['memory_usage']])
        
        print(f"\nGPU MEMORY STATISTICS:")
        print(f"  Average allocated: {avg_mem_alloc:.1f} MB")
        print(f"  Peak memory: {peak_mem:.1f} MB")
        
        # Consistency
        print(f"\nCONSISTENCY:")
        print(f"  Variance: ±{std_time/avg_time*100:.1f}%")
        
        if std_time/avg_time > 0.2:
            print("  WARNING: High variance detected - possible intermittent synchronizations")
        
        # Performance flags
        print(f"\nPERFORMANCE INDICATORS:")
        
        if avg_time/avg_steps > 0.05:  # >50ms per step
            print("  WARNING: Each step takes >50ms - consider optimization")
        
        if avg_mem_alloc > 1000:
            print(f"  WARNING: High memory usage ({avg_mem_alloc:.0f}MB) - possible memory pressure")
        
        # Store results
        self.results = results
        
        print("\n" + "="*80)
        print("GPU DIAGNOSTIC COMPLETED")
        print("="*80)
        
        return results
    
    def quick_gpu_check(self, max_steps=50):
        """
        Quick GPU check - singola run per valutazione rapida
        
        Args:
            max_steps: Numero massimo di step
        
        Returns:
            dict: Risultati rapidi
        """
        print("\n" + "="*80)
        print("QUICK GPU CHECK")
        print("="*80)
        
        # Clean cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Build system
        print("Building GPU system...")
        SNPS_exact_csv()
        snps = SNPSystem()
        snps.load_neurons_from_csv("csv/" + Config.CSV_EXACT_NAME)
        
        msnps = MatrixExecutorExact.translate_to_matrix(snps, device='gpu')
        msnps.loadImages(self.x_train)
        
        if max_steps:
            original_max = msnps.max_steps
            msnps.max_steps = max_steps
        
        # Execute
        print(f"Executing {max_steps if max_steps else 'all'} steps...")
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        msnps.execute(verbose=False)
        
        end_event.record()
        torch.cuda.synchronize()
        exec_time = start_event.elapsed_time(end_event) / 1000
        
        # Results
        print(f"\nRESULTS:")
        print(f"  Steps executed: {msnps.t_step}")
        print(f"  Total time: {exec_time:.3f}s")
        print(f"  Time per step: {exec_time/msnps.t_step*1000:.2f}ms" if msnps.t_step > 0 else "  Time per step: N/A")
        
        # Memory
        mem_alloc = torch.cuda.memory_allocated() / 1024**2
        mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  GPU memory allocated: {mem_alloc:.1f} MB")
        print(f"  Peak GPU memory: {mem_peak:.1f} MB")
        
        # Quick assessment
        print(f"\nQUICK ASSESSMENT:")
        time_per_step_ms = exec_time/msnps.t_step*1000 if msnps.t_step > 0 else 0
        
        if time_per_step_ms < 10:
            print("  EXCELLENT: Fast execution (<10ms per step)")
        elif time_per_step_ms < 50:
            print("  GOOD: Acceptable performance (<50ms per step)")
        elif time_per_step_ms < 100:
            print("  MODERATE: Consider optimization (>50ms per step)")
        else:
            print("  SLOW: Significant optimization needed (>100ms per step)")
            print("    -> Run diagnose_gpu_profiling() to identify bottlenecks")
        
        # Restore
        if max_steps:
            msnps.max_steps = original_max
        
        del msnps
        torch.cuda.empty_cache()
        
        return {
            'steps': msnps.t_step,
            'total_time': exec_time,
            'time_per_step_ms': time_per_step_ms,
            'memory_mb': mem_alloc,
            'peak_memory_mb': mem_peak
        }
    
    def save_results(self, filename="gpu_diagnostic_results.json"):
        """
        Salva i risultati della diagnostica in file JSON
        """
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")


# =========================================================
# MAIN
# =========================================================

def main():
    """
    Funzione principale per eseguire la diagnostica GPU
    """

    database("digit") #can be digit, flower
    Config.MODE = "CNN" 
    Config.compute_k_range()
    # Load your data here - ADAPT THIS TO YOUR CODE
    print("Loading data...")
    
    from sps.digit_image import get_mnist_data
    x_train, y_train, x_test, y_test = get_mnist_data()
    
    # Create diagnostic instance
    diagnostic = MSNPGPUDiagnostic(x_train)
    
    # Option 1: Quick check (fastest)
    print("\nRunning quick GPU check...")
    quick_results = diagnostic.quick_gpu_check(max_steps=50)
    
    # Option 2: GPU performance diagnostic (multiple runs)
    print("\nRunning GPU performance diagnostic...")
    gpu_results = diagnostic.diagnose_gpu_only(num_runs=3, max_steps=100)
    
    # Option 3: Full profiling with detailed analysis
    print("\nRunning full GPU profiling...")
    profile_stats, prof = diagnostic.diagnose_gpu_profiling(max_steps=100)
    
    # Save results
    diagnostic.save_results()
    
    return diagnostic


if __name__ == "__main__":
    main()