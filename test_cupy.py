import cupy as cp
print(f"CuPy Version: {cp.__version__}")
print(f"CUDA disponibile: {cp.cuda.is_available()}")
if cp.cuda.is_available():
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"CUDA Runtime Version: {cp.cuda.runtime.runtimeGetVersion()}")