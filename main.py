from sps import edge_detection, med_mnist, other_networks, cnn, handle_image, cnn , med_mnist2
from sps.config import Config, configure
import numpy as np



configure("quantized") #can be binarized, quantized, edge, cnn, temporal

#med_mnist.launch_binarized_SNPS()
med_mnist.launch_quantized_SNPS()
#med_mnist2.launch_SNPS()
#edge_detection.launch_gray_SNPS()

#OtherNetworks.compute_extended()
#OtherNetworks.compute_divisible_3()
#OtherNetworks.compute_gen_even()