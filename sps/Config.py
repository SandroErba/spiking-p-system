# DATABASE
IMG_SHAPE = 28 #ipotizing squared shape images 28
BLOCK_SHAPE = 4
CLASSES = 8 #8
THRESHOLD = 128 #128

# SPIKING NEURAL P SYSTEM
NEURONS_LAYER1 = int(IMG_SHAPE ** 2) #784
NEURONS_LAYER2 = int((IMG_SHAPE / BLOCK_SHAPE) ** 2) #49
NEURONS_LAYER1_2 = int(NEURONS_LAYER1 + NEURONS_LAYER2) #833
NEURONS_TOTAL = NEURONS_LAYER1_2 + CLASSES # 841

# PARAMETER TUNING
TRAIN_SIZE = 5000
TEST_SIZE = 1000
PRUNING_PERC = 0.4
INHIBIT_PERC = 0.3
POSITIVE_REINFORCE = CLASSES - 1
NEGATIVE_PENALIZATION = 1

# ENERGY COSTS
WORST_REGEX = 100
EXPECTED_REGEX = 10
EXPECTED_SPIKE = 0.5

# STRING
CSV_NAME = "neurons784image.csv"
CSV_NAME_PRUNED = "neurons784image_pruned.csv"
INPUT_TYPE = "image_spike_train"

"""
PRUNING_PERC = 0.2
INHIBIT_PERC = 0.3:
    Top-1 accuracy: 27.1 %
    Top-3 accuracy: 60.5 %
    
    Per-class accuracy:
      Class 0: 7.59% accuracy over 73 instances
      Class 1: 37.10% accuracy over 186 instances
      Class 2: 10.95% accuracy over 86 instances
      Class 3: 67.05% accuracy over 176 instances
      Class 4: 3.74% accuracy over 73 instances
      Class 5: 3.66% accuracy over 82 instances
      Class 6: 19.95% accuracy over 195 instances
      Class 7: 18.93% accuracy over 129 instances
      
---------

PRUNING_PERC = 0.5
INHIBIT_PERC = 0:
    Top-1 accuracy: 27.3 %
    Top-3 accuracy: 63.0 %
    
    Per-class accuracy:
      Class 0: 17.70% accuracy over 73 instances
      Class 1: 40.56% accuracy over 186 instances
      Class 2: 8.81% accuracy over 86 instances
      Class 3: 64.77% accuracy over 176 instances
      Class 4: 1.97% accuracy over 73 instances
      Class 5: 2.44% accuracy over 82 instances
      Class 6: 20.05% accuracy over 195 instances
      Class 7: 17.50% accuracy over 129 instances
      
-------

PRUNING_PERC = 0.4
INHIBIT_PERC = 0.3
    Top-1 accuracy: 27.0 %
    Top-3 accuracy: 60.0 %
    
    Per-class accuracy:
      Class 0: 7.96% accuracy over 73 instances
      Class 1: 33.26% accuracy over 186 instances
      Class 2: 6.81% accuracy over 86 instances
      Class 3: 67.61% accuracy over 176 instances
      Class 4: 4.37% accuracy over 73 instances
      Class 5: 8.32% accuracy over 82 instances
      Class 6: 27.77% accuracy over 195 instances
      Class 7: 13.40% accuracy over 129 instances
"""