import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from sps import edge_detection, med_mnist, other_networks, cnn, med_image, cnn, flower_image, digit_image
from sps.config import Config, configure, database

t=time.time()
database("digit") #can be digit, flower
configure("cnn") #can be binarized, quantized, edge, cnn
cnn.launch_28_CNN_SNPS()

print("Elapsed time:", time.time()-t)


#med_mnist.launch_quantized_SNPS()
#digit_image.launch_gray_SNPS()
#med_mnist.launch_binarized_SNPS()
#edge_detection.launch_gray_SNPS()
#other_networks.compute_extended()
#other_networks.compute_divisible_3()
#other_networks.compute_gen_even()