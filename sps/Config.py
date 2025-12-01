# IMAGES
IMG_SHAPE = 28 #ipotizing squared shape images of 28 pixels
BLOCK_SHAPE = 2 #the size of the window block for the second layer
CLASSES = 8
THRESHOLD = 128 #128 higher Thr -> more spike - with higher threshold I get better performance, but more unbalanced classes -> !shortcut!
INVERT = True #invert or not invert the spike in the starting images (0...4) -> (4...0)
QUANTIZATION = True
Q_RANGE = 4 # the range of quantization, it works on images, rules and tuning TODO generalize the code with it
# TODO NOW: search for an easy CNN and try to recreate it -> https://pythonguides.com/simple-mnist-convnet-keras/
# TODO for B: counting the number of activated rules and tune in this way can be really wrong
# TODO i can try with dropout during training

# PARAMETER TUNING
TRAIN_SIZE = 1000
TEST_SIZE = 1000
PRUNING_PERC = 0.3
INHIBIT_PERC = 0.2
POSITIVE_REINFORCE = CLASSES - 1
NEGATIVE_PENALIZATION = 1

# RULE TUNING
WHITE_HOLE = True #After a rule application, all the remaining spikes are deleted

# NUMBER OF NEURONS
NEURONS_LAYER1 = int(IMG_SHAPE ** 2) #784
NEURONS_LAYER2 = int((IMG_SHAPE / BLOCK_SHAPE) ** 2) #49
NEURONS_LAYER1_2 = int(NEURONS_LAYER1 + NEURONS_LAYER2) #833
NEURONS_TOTAL = NEURONS_LAYER1_2 + CLASSES # 841

# ENERGY COSTS
WORST_REGEX = 100
EXPECTED_REGEX = 10
EXPECTED_SPIKE = 0.5

# SEGMENTATION
KERNEL_SHAPE = 2
KERNEL_NUMBER = 6
SEGMENTED_SHAPE = IMG_SHAPE - KERNEL_SHAPE + 1

# STRING
CSV_NAME_Q = "SNPS_quantize.csv"
CSV_NAME_B = "SNPS_binarize.csv"
CSV_NAME_Q_PRUNED = "SNPS_quantize_pruned.csv"
CSV_NAME_B_PRUNED = "SNPS_binarize_pruned.csv"
CSV_KERNEL_NAME = "SNPS_kernel.csv"
