"""How to read the output table:
r: rule applied. r:7a+2;3->0;1 mean "using 3 charge, 0 spike are fired, with 1 delay". ! = firing rules.
1a is the condition part that follows: if c %7 == 2, the rule applies
c: internal charge of neuron. i:x(y) means "x charge received from neuron y" """

from sps import edge_detection, med_mnist, other_networks, cnn, handle_image, cnn
from sps.config import Config, configure

#CNN.launch_CNN_SNPS()
configure("quantized")
print(Config.QUANTIZATION)
med_mnist.launch_quantized_SNPS()

#EdgeDetection.launch_gray_SNPS() #TRAIN_SIZE < 30, INVERT = False, QUANTIZATION = False

#OtherNetworks.compute_extended()

#OtherNetworks.compute_divisible_3()

#OtherNetworks.compute_gen_even()