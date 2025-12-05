class Config:
    # IMAGES
    IMG_SHAPE = 28 #ipotizing squared shape images of 28 pixels
    CLASSES = 8
    BLOCK_SHAPE = None
    THRESHOLD = None


    INVERT = True #invert or not invert the spike in the starting images (0...4) -> (4...0)
    QUANTIZATION = None
    Q_RANGE = 4 # the range of quantization, it works on images, rules and tuning TODO generalize the code with it

    # TODO for B: counting the number of activated rules and tune in this way can be really wrong
    # TODO i can try with dropout during training

    # PARAMETER TUNING
    TRAIN_SIZE = 1000
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

    #if mode == "cnn":