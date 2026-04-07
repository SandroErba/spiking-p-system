import time
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import  other_networks, cnn, flower_image, digit_image, handle_csv
from sps.config import Config, database
from sps.m_matrix_executor import MatrixExecutor
from sps.m_snp_system import MSNPSystem
from sps.snp_system import SNPSystem

database("digit") #can be digit, flower
Config.MODE = "halting" #set the mode of the P system: can be cnn (default), generative, halting
Config.compute_k_range()
Config.WHITE_HOLE = True 
cnn.test_launch_mnist_cnn()
snps = SNPSystem(0,100,True)
snps.load_neurons_from_csv("csv/" + "SNPS_cnn.csv")
msnp = MatrixExecutor().translate_to_matrix(snps)
print(msnp)

#other_networks.compute_extended() #require halting mode
#other_networks.compute_divisible_3() #require halting mode
#other_networks.compute_gen_even() #require generative mode
#other_networks.prova() #require halting mode

# print("#"*50)
# snps = SNPSystem(0,100,True)
# snps.load_neurons_from_csv("csv/" + "neuronsDiv3.csv")
#snps.spike_train = [1, 0, 0, 0, 0, 0, 0, 1]
#msnp = MatrixExecutor().translate_to_matrix(snps)
#print(msnp)
#msnp.execute(True)
