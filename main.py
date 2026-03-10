import time
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import  other_networks, cnn, flower_image, digit_image, handle_csv, med_image
from sps.config import Config, database


database("bloodmnist") #can be digit, flower
#Config.MODE = "generative" #set the mode of the P system: can be cnn (default), generative, halting
Config.compute_k_range()

# ---------------- BLOCK 5 (10 TEST-ONLY RUNS) ----------------
# Obiettivo: migliorare molto LR/SVM senza retrain.
# IMPORTANTE: i parametri di train-key restano fissi per riuso cache:
# DATABASE, TRAIN_SIZE, CSV_NAME, SVM_C, Q_RANGE, KERNEL_NUMBER.
Config.SVM_C = 3.0

# Test 1 - baseline attuale migliore (veloce)
Config.ALPHA_METHOD = 2
Config.DISCRETIZE_METHOD = 1
Config.DISC_RANGE = 2
Config.QUANTIZE_METHOD = 2
Config.M_THRESHOLD = 0.7
cnn.launch_mnist_cnn()

# Test 2 - soglia un po' piu alta
Config.ALPHA_METHOD = 2
Config.DISCRETIZE_METHOD = 1
Config.DISC_RANGE = 2
Config.QUANTIZE_METHOD = 2
Config.M_THRESHOLD = 0.8
cnn.launch_mnist_cnn()

# Test 3 - soglia alta
Config.ALPHA_METHOD = 2
Config.DISCRETIZE_METHOD = 1
Config.DISC_RANGE = 2
Config.QUANTIZE_METHOD = 2
Config.M_THRESHOLD = 0.9
cnn.launch_mnist_cnn()

# Test 4 - soglia molto alta
Config.ALPHA_METHOD = 2
Config.DISCRETIZE_METHOD = 1
Config.DISC_RANGE = 2
Config.QUANTIZE_METHOD = 2
Config.M_THRESHOLD = 1.0
cnn.launch_mnist_cnn()

# Test 5 - soglia intermedia
Config.ALPHA_METHOD = 2
Config.DISCRETIZE_METHOD = 1
Config.DISC_RANGE = 2
Config.QUANTIZE_METHOD = 2
Config.M_THRESHOLD = 0.6
cnn.launch_mnist_cnn()

# Test 6 - confronto col miglior TWN storico
Config.ALPHA_METHOD = 2
Config.DISCRETIZE_METHOD = 1
Config.DISC_RANGE = 2
Config.QUANTIZE_METHOD = 3
Config.M_THRESHOLD = 0.7
cnn.launch_mnist_cnn()

# Test 7 - quantize percentile: piu sparsita
Config.ALPHA_METHOD = 2
Config.DISCRETIZE_METHOD = 1
Config.DISC_RANGE = 2
Config.QUANTIZE_METHOD = 1
Config.M_SPARSITY = 0.6
Config.M_POSITIVE = 0.2
cnn.launch_mnist_cnn()

# Test 8 - quantize percentile: molto sparso
Config.ALPHA_METHOD = 2
Config.DISCRETIZE_METHOD = 1
Config.DISC_RANGE = 2
Config.QUANTIZE_METHOD = 1
Config.M_SPARSITY = 0.7
Config.M_POSITIVE = 0.15
cnn.launch_mnist_cnn()

# Test 9 - quantize percentile: meno sparso
Config.ALPHA_METHOD = 2
Config.DISCRETIZE_METHOD = 1
Config.DISC_RANGE = 2
Config.QUANTIZE_METHOD = 1
Config.M_SPARSITY = 0.4
Config.M_POSITIVE = 0.3
cnn.launch_mnist_cnn()

# Test 10 - cambio solo alpha (potenziale boost SVM)
Config.ALPHA_METHOD = 1
Config.DISCRETIZE_METHOD = 1
Config.DISC_RANGE = 2
Config.QUANTIZE_METHOD = 2
Config.M_THRESHOLD = 0.7
cnn.launch_mnist_cnn()



#edge_detection.launch_gray_SNPS()

#other_networks.compute_extended() #require halting mode
#other_networks.compute_divisible_3() #require halting mode
#other_networks.compute_gen_even() #require generative mode