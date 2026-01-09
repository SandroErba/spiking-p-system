from sps import edge_detection, med_mnist, other_networks, cnn, med_image, cnn, flower_image, digit_image
from sps.config import Config, configure


configure("quantized") #can be binarized, quantized, edge, cnn
configure("digit") #can be medmnist, digit, flower102
print(Config.NEURONS_LAYER3, Config.NEURONS_TOTAL, Config.NEURONS_LAYER1_2)
digit_image.launch_gray_SNPS()

#cnn.launch_CNN_SNPS()

#med_mnist.launch_binarized_SNPS()
#med_mnist.launch_quantized_SNPS()

#edge_detection.launch_gray_SNPS()

#other_networks.compute_extended()
#other_networks.compute_divisible_3()
#other_networks.compute_gen_even()