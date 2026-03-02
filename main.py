import time
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import  other_networks, cnn, cnn, flower_image, digit_image, handle_csv
from sps.config import Config, database


database("digit") #can be digit, flower
#Config.MODE = "generative" #set the mode of the P system: can be cnn (default), generative, halting
Config.compute_k_range()


t=time.time()
accuracy = cnn.launch_28_CNN_SNPS()

"""handle_csv.log_experiment(
    params={
        "train size": Config.TRAIN_SIZE,
        "test size": Config.TEST_SIZE,
        "q range": Config.Q_RANGE,
        "perceptron sparsity": Config.SPARSITY,
        "perceptron positive": Config.POSITIVE,
        "perceptron lr": Config.LR,
        "database": Config.DATABASE,
        "kernel number": Config.KERNEL_NUMBER
    },
    metrics={
        "accuracy": accuracy,
        "time": time.time()-t
    }
)"""



#edge_detection.launch_gray_SNPS()

#other_networks.compute_extended() #require halting mode
#other_networks.compute_divisible_3() #require halting mode
#other_networks.compute_gen_even() #require generative mode