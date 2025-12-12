class Config:
    # IMAGES
    IMG_SHAPE = 28 #ipotizing squared shape images of 28 pixels
    CLASSES = 8
    BLOCK_SHAPE = None
    THRESHOLD = 128
    TEMPORAL_THRESHOLD_LEVELS = None #[ 75, 120,  180]
    MAX_TIME_STEPS = None
    NUM_INPUT_LEVELS = None


    INVERT = True #invert or not invert the spike in the starting images (0...4) -> (4...0)
    QUANTIZATION = None
    TEMPORAL = None
    Q_RANGE = 4 # the range of quantization, it works on images, rules and tuning TODO generalize the code with it

    # TODO for B: counting the number of activated rules and tune in this way can be really wrong
    # TODO i can try with dropout during training

    # PARAMETER TUNING
    TRAIN_SIZE = 100
    TEST_SIZE = 1000
    PRUNING_PERC = 0.3
    INHIBIT_PERC = 0.2
    POSITIVE_REINFORCE = None
    NEGATIVE_PENALIZATION = None


    # RULE TUNING
    WHITE_HOLE = True #After a rule application, all the remaining spikes are deleted

    # NUMBER OF NEURONS
    NEURONS_LAYER1 = None
    NEURONS_LAYER2 = None
    NEURONS_LAYER2_PER_LEVEL = None
    NEURONS_LAYER1_2 = None
    NEURONS_TOTAL = None

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
    CSV_NAME_CNN = "SNPS_CNN.csv"
    CSV_NAME_T = "SNPS_temporal.csv"
    CSV_NAME_T_PRUNED = "SNPS_temporal_pruned.csv"

def configure(mode):
    if mode == "quantized":
        print("SETTING")
        Config.BLOCK_SHAPE = 2 #the size of the window block for the second layer
        Config.QUANTIZATION = True

        Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2) #784
        Config.NEURONS_LAYER2 = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) #49
        Config.NEURONS_LAYER1_2 = int(Config.NEURONS_LAYER1 + Config.NEURONS_LAYER2) #833
        Config.NEURONS_TOTAL = Config.NEURONS_LAYER1_2 + Config.CLASSES # 841

    if mode == "binarized":
        Config.BLOCK_SHAPE = 4 #the size of the window block for the second layer
        Config.THRESHOLD = 128 #128 higher Thr -> more spike - with higher threshold I get better performance, but more unbalanced classes -> !shortcut!
        Config.QUANTIZATION = False

        Config.POSITIVE_REINFORCE = Config.CLASSES - 1
        Config.NEGATIVE_PENALIZATION = 1

        Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2) #784
        Config.NEURONS_LAYER2 = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) #49
        Config.NEURONS_LAYER1_2 = int(Config.NEURONS_LAYER1 + Config.NEURONS_LAYER2) #833
        Config.NEURONS_TOTAL = Config.NEURONS_LAYER1_2 + Config.CLASSES # 841

    if mode == "edge" :
        Config.TRAIN_SIZE = 30
        Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2)
        Config.layer2_size_per_kernel = int (Config.SEGMENTED_SHAPE ** 2)
        Config.NEURONS_LAYER2 = Config.layer2_size_per_kernel * Config.KERNEL_NUMBER
        Config.NEURONS_LAYER1_2 = Config.NEURONS_LAYER1 + Config.NEURONS_LAYER2

    if mode == "temporal":
        Config.BLOCK_SHAPE = 3
        if(Config.BLOCK_SHAPE == 3):
            Config.IMG_SHAPE = 27
        Config.TEMPORAL_THRESHOLD_LEVELS = [50, 90, 150,  190] #[50, 100, 150, 200]  #    #[50, 100, 150, 200]    #[65, 110, 150, 190]   #[50, 90, 130, 170, 210]     #     #[ 75, 120,  180]     #[75, 120, 180]    #[74, 118, 154, 184, 211, 237]     #[64, 94, 118, 139, 157, 174, 191, 207, 221, 240]     #[213, 170, 128, 85, 43] # levels
        Config.MAX_TIME_STEPS = Config.TEMPORAL_THRESHOLD_LEVELS.__len__() + 1 #5
        Config.NUM_INPUT_LEVELS = Config.MAX_TIME_STEPS 
        Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2) * Config.NUM_INPUT_LEVELS #784
        Config.NEURONS_LAYER2_PER_LEVEL = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) #49
        Config.NEURONS_LAYER2 = Config.NEURONS_LAYER2_PER_LEVEL * Config.MAX_TIME_STEPS #245
        Config.NEURONS_LAYER1_2 = int(Config.NEURONS_LAYER1 + Config.NEURONS_LAYER2) #833
        Config.NEURONS_TOTAL = Config.NEURONS_LAYER1_2 + Config.CLASSES # 841

        # Config.POSITIVE_REINFORCE = Config.CLASSES * 2.5
        # Config.NEGATIVE_PENALIZATION = 0.5
        # Config.PRUNING_PERC = 0.4
        # Config.INHIBIT_PERC = 0.15
        # Config.TEMPORAL = True

        Config.POSITIVE_REINFORCE = Config.CLASSES * 2.4
        Config.NEGATIVE_PENALIZATION = 0.7
        Config.PRUNING_PERC = 0.5
        Config.INHIBIT_PERC = 0.2
        Config.TEMPORAL = True


    #if mode == "cnn":