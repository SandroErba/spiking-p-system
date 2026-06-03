import os
os.environ['CUDA_PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
os.environ['PATH'] = os.environ['CUDA_PATH'] + r'\bin;' + os.environ.get('PATH', '')


import cupy as cp
import numpy as np

print("=== INFORMAZIONI SISTEMA ===")
print(f"CuPy version: {cp.__version__}")
print(f"CUDA disponibile: {cp.cuda.is_available()}")
print(f"Numero GPU: {cp.cuda.runtime.getDeviceCount()}")

if cp.cuda.is_available():
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    print(f"GPU: {props['name'].decode()}")
    print(f"Memoria: {props['totalGlobalMem'] / 1024**3:.1f} GB")
    print(f"Compute Capability: {props['major']}.{props['minor']}")
    
    # Test memoria
    print("\n=== TEST BANDWIDTH ===")
    size = 1000000
    a = cp.random.rand(size)
    b = cp.random.rand(size)
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    c = a + b
    end.record()
    end.synchronize()
    print(f"Addizione {size} elementi: {cp.cuda.get_elapsed_time(start, end):.2f} ms")
    
    # Test diverse dimensioni
    print("\n=== TEST MOLTIPLICAZIONI MATRICI ===")
    for size in [1000, 2000, 5000, 10000]:
        a = cp.random.rand(size, size)
        b = cp.random.rand(size, size)
        
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        c = cp.dot(a, b)
        end.record()
        end.synchronize()
        
        time_ms = cp.cuda.get_elapsed_time(start, end)
        print(f"{size}x{size}: {time_ms:.2f} ms")