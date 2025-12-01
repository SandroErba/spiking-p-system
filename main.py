"""How to read the output table:
r: rule applied. r:7a+2;3->0;1 mean "using 3 charge, 0 spike are fired, with 1 delay". ! = firing rules.
1a is the condition part that follows: if c %7 == 2, the rule applies
c: internal charge of neuron. i:x(y) means "x charge received from neuron y" """

from sps import EdgeDetection, MedMnist, OtherNetworks, Config


MedMnist.launch_quantized_SNPS() if Config.QUANTIZATION else MedMnist.launch_binarized_SNPS()


#EdgeDetection.launch_gray_SNPS() #Use Config.TRAIN_SIZE < 30 for a good visualization

#OtherNetworks.compute_extended()

#OtherNetworks.compute_divisible_3()

#OtherNetworks.compute_gen_even()