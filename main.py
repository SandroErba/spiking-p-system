import time
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import  other_networks, cnn, flower_image, digit_image, handle_csv
from sps.config import Config, database


database("digit") #can be digit, flower
#Config.MODE = "generative" #set the mode of the P system: can be cnn (default), generative, halting
Config.compute_k_range()


#t=time.time()
svm_accuracy, lr_accuracy = cnn.launch_mnist_cnn()

#handle_csv.save_results(svm_accuracy, lr_accuracy, time.time()-t) #save the results in the results.csv file as JSON




#edge_detection.launch_gray_SNPS()

#other_networks.compute_extended() #require halting mode
#other_networks.compute_divisible_3() #require halting mode
#other_networks.compute_gen_even() #require generative mode