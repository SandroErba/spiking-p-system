from sps import edge_detection, med_mnist, other_networks, cnn, handle_image, cnn
from sps.config import Config, configure


configure("quantized") #can be binarized, quantized, edge, cnn

#cnn.launch_CNN_SNPS()

#med_mnist.launch_binarized_SNPS()
med_mnist.launch_quantized_SNPS()
#med_mnist.launch_cnn_classification_SNPS()

#edge_detection.launch_gray_SNPS()

#OtherNetworks.compute_extended()
#OtherNetworks.compute_divisible_3()
#OtherNetworks.compute_gen_even()


