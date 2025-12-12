from sps import edge_detection, med_mnist, other_networks, cnn, handle_image, cnn
from sps.config import Config, configure

<<<<<<< HEAD
from sps import edge_detection, med_mnist, other_networks, cnn, handle_image, cnn, edge_detection2, med_mnist2
from sps.config import Config, configure

#CNN.launch_CNN_SNPS()
# configure("quantized")
# print("quantized " + str(Config.QUANTIZATION))
# med_mnist.launch_quantized_SNPS()

#Config.TRAIN_SIZE = 30
#edge_detection.kernel_SNPS_csv()
#edge_detection.launch_gray_SNPS() #TRAIN_SIZE < 30, INVERT = False, QUANTIZATION = False

# configure("temporal")
# print("temporal " + str(Config.TEMPORAL) )
# med_mnist2.launch_SNPS()

edge_detection2.kernel_SNPS_csv()
edge_detection2.launch_gray_SNPS()

#OtherNetworks.compute_extended()

#OtherNetworks.compute_divisible_3()
=======

configure("quantized") #can be binarized, quantized, edge, cnn

#cnn.launch_CNN_SNPS()

#med_mnist.launch_binarized_SNPS()
med_mnist.launch_quantized_SNPS()
>>>>>>> 2c97215ec8e5ed5131013ed1d66014bbead5d477

#edge_detection.launch_gray_SNPS()

#OtherNetworks.compute_extended()
#OtherNetworks.compute_divisible_3()
#OtherNetworks.compute_gen_even()