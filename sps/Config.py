# DATABASE
IMG_SHAPE = 28 #ipotizing squared shape images of 28 pixels
BLOCK_SHAPE = 4 #the size of the window block for the second layer
CLASSES = 8
THRESHOLD = 128 #128 higher Thr -> more spike - with higher threshold I get better performance, but more unbalanced classes -> !shortcut!

# PARAMETER TUNING
TRAIN_SIZE = 1000
TEST_SIZE = 1000
PRUNING_PERC = 0.3
INHIBIT_PERC = 0.2
POSITIVE_REINFORCE = CLASSES - 1
NEGATIVE_PENALIZATION = 1

# RULE TUNING
WHITE_HOLE = True #After a rule application, all the remaining spikes are deleted
EXTENDED = False #The rules have 6 parameters, and the 6th is the number of spike sent, that can be more than 1

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

#TODO !!!when i added the csv/ , the performance has increased but with different classes. see 11/17 commit
# change threshold; count the number of activated rules and tune in this way can be really wrong

# STRING
CSV_NAME = "SNPS_classification.csv"
CSV_NAME_PRUNED = "SNPS_classification_pruned.csv"
CSV_KERNEL_NAME = "SNPS_kernel.csv"
