'''How to read the output table:
r: rule applied. r:7a+2;3->0;1 mean "using 3 charge, 0 spike are fired, with 1 delay". ! = firing rules.
1a is the condition part that follows: if c %7 == 2, the rule applies
c: internal charge of neuron. i:x(y) means "x charge received from neuron y"
'''

from sps import EdgeDetection, MedMnist
#MedMnist.launch_SNPS()
EdgeDetection.launch_gray_SNPS()