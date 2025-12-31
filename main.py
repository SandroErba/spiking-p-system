from sps import edge_detection, med_mnist, other_networks, cnn, handle_image, cnn
from sps.config import Config, configure
import numpy as np
from sps import edge_detection, med_mnist, other_networks, cnn, handle_image, cnn, med_mnist2   , edge_detection_mio
from sps.config import Config, configure

#CNN.launch_CNN_SNPS()
# configure("quantized")
# print("quantized " + str(Config.QUANTIZATION))
# med_mnist.launch_quantized_SNPS()

#Config.TRAIN_SIZE = 30
#edge_detection.kernel_SNPS_csv()
#edge_detection.launch_gray_SNPS() #TRAIN_SIZE < 30, INVERT = False, QUANTIZATION = False



#configure("edge_map")
# edge_detection_mio.generate_manual_csv()
# edge_detection_mio.launch_gray_SNPS()

#OtherNetworks.compute_extended()

#OtherNetworks.compute_divisible_3()

configure("temporal") #can be binarized, quantized, edge, cnn, temporal

#med_mnist.launch_binarized_SNPS()
#med_mnist.launch_quantized_SNPS()
med_mnist2.launch_SNPS()
#edge_detection.launch_gray_SNPS()

#OtherNetworks.compute_extended()
#OtherNetworks.compute_divisible_3()
#OtherNetworks.compute_gen_even()