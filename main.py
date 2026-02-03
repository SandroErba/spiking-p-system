import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sps import edge_detection, med_mnist, other_networks, cnn, med_image, cnn, flower_image, digit_image
from sps.config import Config, configure
#from sps.digit_image import get_28_digit_data

#digit_image.get_28_digit_data()
configure("cnn") #can be binarized, quantized, edge, cnn
#med_mnist.launch_quantized_SNPS()
configure("digit") #can be digit, flower102
cnn.launch_28_CNN_SNPS()
#digit_image.launch_gray_SNPS()

#med_mnist.launch_binarized_SNPS()


#edge_detection.launch_gray_SNPS()

#other_networks.compute_extended()
#other_networks.compute_divisible_3()
#other_networks.compute_gen_even()