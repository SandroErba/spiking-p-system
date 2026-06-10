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


# Usalo all'inizio del tuo script
reset_gpu_for_rerun()

database("digit") #can be digit, flower
# Config.MODE = "CNN" #set the mode of the P system: can be cnn (default), generative, halting
Config.compute_k_range()

#snps = network.create_exact_csv()
system = "SNPSystem"
network.launch_mnist(system)

system = "MSNPSystemExactGPU"
network.launch_mnist(system)



def reset_gpu_for_rerun():
    """Prepara la GPU per una nuova esecuzione."""
    
    # Elimina variabili globali se necessario
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except:
            pass
    
    # Pulizia aggressiva
    gc.collect()
    torch.cuda.empty_cache()
    
    # Sincronizza GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

#MSNPSystemDivModGPU, MSNPSystemExactGPU, SNPSystem

#network.launch_mnist_from_csv("SNPS_cnn_external.csv")

#Config.NUM_LAYERS = 6
#network.launch_mnist_from_csv("SNPS_deep_cnn.csv")

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


#M = np.array([[-1, 1, 1,0], [-2, 1, 1,0], [1, -1, 1,0],[0,0,-1,1],[0,0,-2,0],[0,0,0,0]])
#ruleVector = np.array([[0,2],[0,2],[0,1],[0,1],[0,2],[0,0]])
#c0 = np.array([2, 1, 1, 0])
#spikingVector = np.array([0, 0, 0, 0, 0, 0])
#netGainVector = np.zeros((4,), dtype=int)
#msnp = MSNPSystem(configurationVector=c0, spikingVector=spikingVector, spikingTransitionMatrix=M, netGainVector=netGainVector, ruleVector=ruleVector, max_steps=100, deterministic=False,applyingRuleVector=np.array([0, 0, 1, 2, 2, 3]), targetVector=np.array([1, 1, 1, 1, 0, 0]))
#msnp.execute(verbose=True)

# M = np.array([[-1,1,1,0],[1,-1,1,1],[0,0,-1,0],[0,0,-3,-2]])
# ruleVector = np.array([[0,1],[0,1],[0,1],[0,2]])
# c0 = np.array([1, 1, 0,1])
# spikingVector = np.array([0,0,0,0])
# netGainVector = np.zeros((4,), dtype=int)
# msnp = MSNPSystem(configurationVector=c0, spikingVector=spikingVector, spikingTransitionMatrix=M, netGainVector=netGainVector, ruleVector=ruleVector, max_steps=50, deterministic=True,applyingRuleVector=np.array([0,1,2,3]), targetVector=np.array([1,1,0,-1]))
# msnp.execute(verbose=True)

# M = np.array([[-1,1,1,0,0,0],[1,-1,1,0,0,0],[0,0,-1,0,0,0],[0,0,-3,-6,0,0],[0,0,0,3,-3,3],[0,0,0,0,3,-3]])
# ruleVector = np.array([[0,1],[0,1],[0,0],[0,6],[0,3],[0,3]])
# c0 = np.array([1,1,0,3,3,3])
# spikingVector = np.array([0,0,0,0,0,0])
# netGainVector = np.zeros((6,), dtype=int)
# msnp = MSNPSystem(configurationVector=c0, spikingVector=spikingVector, spikingTransitionMatrix=M, netGainVector=netGainVector, ruleVector=ruleVector, max_steps=10, deterministic=True,applyingRuleVector=np.array([0,1,2,3,4,5]), targetVector=np.array([1,1,0,-3,3,3]))
# msnp.execute(verbose=True)

# M = np.array([[-1,1,1,0,0],[1,-1,1,1,0],[0,0,-2,0,2],[0,0,-3,0,1],[0,0,1,-2,0]])
# ruleVector = np.array([[0,1],[0,1],[0,2],[0,3],[0,2]])
# c0 = np.array([1,1,0,1,0])
# spikingVector = np.array([0,0,0,0,0])
# netGainVector = np.zeros((5,), dtype=int)
# applyingRuleVector = np.array([0,1,2,2,3])
# targetVector = np.array([1,1,2,1,1])
# msnp = MSNPSystem(configurationVector=c0, spikingVector=spikingVector, spikingTransitionMatrix=M, netGainVector=netGainVector, ruleVector=ruleVector, max_steps=20, deterministic=True,applyingRuleVector=applyingRuleVector, targetVector=targetVector)
# msnp.execute(verbose=True)

