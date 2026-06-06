import time
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from datetime import datetime

from sps import  other_networks, network, flower_image, digit_image, med_image, handle_csv
from sps.config import Config, database

#Config.MODE = "generative" #set the mode of the P system: can be cnn (default), generative, halting
database("digit") #can be digit, flower, tissuemnist, breastmnist, octmnist, bloodmnist, pathmnist,
Config.compute_k_range()

#snps = network.create_exact_csv()
network.launch_mnist()

#network.launch_mnist_from_csv("SNPS_cnn_external.csv")

#Config.NUM_LAYERS = 6
#network.launch_mnist_from_csv("SNPS_deep_cnn.csv")

#Config.MODE = "halting"
#other_networks.compute_extended() #require halting mode
#other_networks.compute_divisible_3() #require halting mode

#Config.MODE = "generative"
#other_networks.compute_gen_even() #require generative mode

#print(f"[{datetime.now()}] computation halted", flush=True)