import time
import os
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import  other_networks, cnn, flower_image, digit_image, handle_csv
from sps.config import Config, database
from sps.m_matrix_executor import MatrixExecutor
from sps.m_snp_system import MSNPSystem
from sps.snp_system import SNPSystem

# database("digit") #can be digit, flower
# Config.MODE = "CNN" #set the mode of the P system: can be cnn (default), generative, halting
# Config.compute_k_range()
Config.WHITE_HOLE = False 
# snps = cnn.test_launch_mnist_cnn()

# msnp = MatrixExecutor().translate_to_matrix(snps)
# msnp.loadImages(snps.spike_train)
# print(msnp.img_spike_train[0])
# print(msnp.img_spike_train.shape)
# msnp.step(verbose=True)
# print("#"*50)
# print(msnp.configurationVector if np.any(msnp.configurationVector) else "All neurons have zero charge")

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


M = np.array([[-1, 1, 1,0], [-2, 1, 1,0], [1, -1, 1,0],[0,0,-1,1],[0,0,-2,0],[0,0,0,0]])
ruleVector = np.array([[0,2],[0,2],[0,1],[0,1],[0,2],[0,0]])
c0 = np.array([2, 1, 1, 0])
spikingVector = np.array([0, 0, 0, 0, 0, 0])
netGainVector = np.zeros((4,), dtype=int)
msnp = MSNPSystem(configurationVector=c0, spikingVector=spikingVector, spikingTransitionMatrix=M, netGainVector=netGainVector, ruleVector=ruleVector, max_steps=100, deterministic=False,applyingRuleVector=np.array([0, 0, 1, 2, 2, 3]), targetVector=np.array([1, 1, 1, 1, 0, 0]))
msnp.execute(verbose=True)