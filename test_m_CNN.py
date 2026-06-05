import time
import os
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import other_networks, cnn, flower_image, digit_image, handle_csv
from sps.config import Config, database
from sps.m_matrix_executor_pytorch import MatrixExecutor
from sps.m_snp_system import MSNPSystem
from sps.m_gpu import MSNPSystemGPU
from sps.snp_system import SNPSystem

print("Testing MSNPSystemGPU with MNIST CNN...")
database("digit")
Config.MODE = "CNN"
Config.compute_k_range()
Config.WHITE_HOLE = True 

snps = cnn.test_launch_mnist_cnn()

# CORREZIONE 1: Chiamata corretta al metodo statico
msnp_gpu = MatrixExecutor.translate_to_matrix(snps)  # Nota: senza ()

msnp_gpu.loadImages(snps.spike_train)

# CORREZIONE 2: Nome variabile corretto
msnp_gpu.step(verbose=True)  # era msnpu_gpu