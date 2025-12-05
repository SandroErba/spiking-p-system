"""How to read the output table:
r: rule applied. r:7a+2;3->0;1 mean "using 3 charge, 0 spike are fired, with 1 delay". ! = firing rules.
1a is the condition part that follows: if c %7 == 2, the rule applies
c: internal charge of neuron. i:x(y) means "x charge received from neuron y" """
import medmnist
import numpy as np
from medmnist import INFO
from sps import EdgeDetection, MedMnist, OtherNetworks, EdgeDetection2, MioMedMnist1, MioMedMnist2, MioMedMnist3, Config, Trova_soglie

#Trova_soglie.find_optimal_thresholds_kmeans(n_thresholds=6)

#MedMnist.launch_SNPS()

#MioMedMnist1.launch_SNPS()

# info = INFO['bloodmnist']
# DataClass = getattr(medmnist, info['python_class'])
# test_dataset = DataClass(split='test', download=True)
    
#     # 2. Definisci QUALI immagini vuoi vedere (es. la 0, la 12 e la 45)   1 3 5 11 15 16 14 26 cl 0 1 3 5 6 2 4
# my_indices = [1, 3, 5, 11, 14, 15, 16, 26]
    
#     # 3. Prendi le soglie dal tuo Config
# my_thresholds = 30 #Config.TEMPORAL_THRESHOLD_LEVELS 
    
#     # 4. CHIAMATA UNICA per visualizzare tutto
# MioMedMnist2.visualize_batch(test_dataset, my_indices, my_thresholds)

MioMedMnist2.launch_SNPS()

#Config.NUM_CHANNELS = 4
#MioMedMnist3.launch_SNPS()

#EdgeDetection.launch_gray_SNPS() #Use Config.TRAIN_SIZE < 30 for a good visualization

#EdgeDetection2.launch_gray_SNPS() #Use Config.TRAIN_SIZE < 30 for a good visualization

#OtherNetworks.compute_divisible_3()

#OtherNetworks.compute_gen_even()