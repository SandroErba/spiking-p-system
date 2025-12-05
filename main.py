"""How to read the output table:
r: rule applied. r:7a+2;3->0;1 mean "using 3 charge, 0 spike are fired, with 1 delay". ! = firing rules.
1a is the condition part that follows: if c %7 == 2, the rule applies
c: internal charge of neuron. i:x(y) means "x charge received from neuron y" """

from sps import edge_detection, med_mnist, other_networks, cnn, handle_image, cnn, edge_detection2, med_mnist2
from sps.config import Config, configure

#CNN.launch_CNN_SNPS()
# configure("quantized")
# print(Config.QUANTIZATION)
#med_mnist.launch_quantized_SNPS()

#Config.TRAIN_SIZE = 30
#edge_detection.launch_gray_SNPS() #TRAIN_SIZE < 30, INVERT = False, QUANTIZATION = False

configure("temporal")
print(Config.TEMPORAL)
med_mnist2.launch_SNPS()

# edge_detection2.kernel_SNPS_csv()
# edge_detection2.launch_gray_SNPS()

#OtherNetworks.compute_extended()

#OtherNetworks.compute_divisible_3()

#OtherNetworks.compute_gen_even()