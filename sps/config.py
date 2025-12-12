class Config:
    MODE = "Default"
    # IMAGES
    IMG_SHAPE = 28 #ipotizing squared shape images of 28 pixels
    CLASSES = 8
    BLOCK_SHAPE = None  #the size of the window block for the second layer
    THRESHOLD = None #128 higher Thr -> more spike - with higher threshold I get better performance, but more unbalanced classes -> !shortcut!

    INVERT = False #invert or not invert the spike in the starting images (0...4) -> (4...0)
    QUANTIZATION = None
    Q_RANGE = None # the range of quantization, it works on images, rules and tuning

    # PARAMETER TUNING
    TRAIN_SIZE = None
    TEST_SIZE = None
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
    KERNEL_SHAPE = None # 2 or 3
    KERNEL_NUMBER = None #TODO not matching with cnn
    SEGMENTED_SHAPE = None # 26

    # STRING
    CSV_NAME = "Null"
    CSV_NAME_PRUNED = "Null"

# TODO use mode in all the code for handle different behaviour, maybe managing input and output type
def configure(mode):
    if mode == "quantized":
        Config.MODE = "quantized"
        Config.BLOCK_SHAPE = 2 #the size of the window block for the second layer
        Config.QUANTIZATION = True
        Config.Q_RANGE = 4 #TODO generalize the code with it
        Config.INVERT = True

        Config.TRAIN_SIZE = 1000
        Config.TEST_SIZE = 1000

        Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2) #784
        Config.NEURONS_LAYER2 = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) #49
        Config.NEURONS_LAYER1_2 = int(Config.NEURONS_LAYER1 + Config.NEURONS_LAYER2) #833
        Config.NEURONS_TOTAL = Config.NEURONS_LAYER1_2 + Config.CLASSES # 841


        Config.CSV_NAME = "SNPS_quantize.csv"
        Config.CSV_NAME_PRUNED = "SNPS_quantize_pruned.csv"

    if mode == "binarized":
        Config.MODE = "binarized"
        Config.BLOCK_SHAPE = 4
        Config.THRESHOLD = 128
        Config.QUANTIZATION = False
        Config.INVERT = True


        Config.TRAIN_SIZE = 1000
        Config.TEST_SIZE = 1000

        Config.POSITIVE_REINFORCE = Config.CLASSES - 1
        Config.NEGATIVE_PENALIZATION = 1

        Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2) #784
        Config.NEURONS_LAYER2 = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) #49
        Config.NEURONS_LAYER1_2 = int(Config.NEURONS_LAYER1 + Config.NEURONS_LAYER2) #833
        Config.NEURONS_TOTAL = Config.NEURONS_LAYER1_2 + Config.CLASSES # 841

        Config.CSV_NAME = "SNPS_binarize.csv"
        Config.CSV_NAME_PRUNED = "SNPS_binarize_pruned.csv"

    if mode == "edge":
        Config.MODE = "edge"
        Config.BLOCK_SHAPE = 4
        Config.THRESHOLD = 128
        Config.INVERT = False
        Config.TRAIN_SIZE = 30
        Config.CSV_NAME = "SNPS_kernel.csv"
        Config.KERNEL_SHAPE = 2
        Config.KERNEL_NUMBER = 6
        Config.SEGMENTED_SHAPE = Config.IMG_SHAPE - Config.KERNEL_SHAPE + 1 # 27

        Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2) #784
        Config.NEURONS_LAYER2 = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) #49
        Config.NEURONS_LAYER1_2 = int(Config.NEURONS_LAYER1 + Config.NEURONS_LAYER2) #833
        Config.NEURONS_TOTAL = Config.NEURONS_LAYER1_2 + Config.CLASSES # 841


    if mode == "cnn":
        Config.MODE = "cnn"
        Config.QUANTIZATION = True
        Config.Q_RANGE = 4
        Config.TRAIN_SIZE = 13
        Config.INVERT = False

        Config.KERNEL_SHAPE = 3
        Config.KERNEL_NUMBER = 1

        Config.SEGMENTED_SHAPE = Config.IMG_SHAPE - Config.KERNEL_SHAPE + 1 # 26
        Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2) #784

        Config.CSV_NAME = "SNPS_cnn.csv"
