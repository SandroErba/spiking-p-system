import time
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import  other_networks, cnn, flower_image, digit_image, handle_csv
from sps.config import Config, database

#Config.MODE = "generative" #set the mode of the P system: can be cnn (default), generative, halting
database("digit") #can be digit, flower, tissuemnist, breastmnist, octmnist, bloodmnist, pathmnist,
Config.compute_k_range()
cnn.launch_mnist_cnn()



#Test per controllare se l'aumento di *10 del test size conferma le performance maggiori oppure sono solo i primi dati a matchare. farlo sulle due con 0.953 e altre a caso


#Config.MODE = "halting"
#other_networks.compute_extended() #require halting mode
#other_networks.compute_divisible_3() #require halting mode

#Config.MODE = "generative"
#other_networks.compute_gen_even() #require generative mode