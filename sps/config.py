class Config:
    MODE = "Default"
    DATABASE = 'medmnist'
    # IMAGES
    IMG_SHAPE = 28 #ipotizing squared shape images of 28 pixels
    CLASSES = 8
    BLOCK_SHAPE = None  #the size of the window block for the second layer
    THRESHOLD = None #128 higher Thr -> more spike - with higher threshold I get better performance, but more unbalanced classes -> !shortcut!

    INVERT = False #invert or not invert the spike in the starting images (0...4) -> (4...0)
    QUANTIZATION = None
    Q_RANGE = None # the range of quantization, it works on images, rules and tuning

    # PARAMETER TUNING
    TRAIN_SIZE = 1000
    TEST_SIZE = 1000
    PRUNING_PERC = 0.2 #0.2
    INHIBIT_PERC = 0.4 #0.3
    POSITIVE_REINFORCE = CLASSES - 1
    NEGATIVE_PENALIZATION = 1

    # ENERGY COSTS
    WORST_REGEX = 100
    EXPECTED_REGEX = 10
    EXPECTED_SPIKE = 0.5

    # SEGMENTATION
    KERNEL_SHAPE = 2
    KERNEL_NUMBER = 6
    SEGMENTED_SHAPE = IMG_SHAPE - KERNEL_SHAPE + 1

    # STRING
    CSV_NAME = "Null"
    CSV_NAME_PRUNED = "Null"

    WHITE_HOLE= True #if true all the internal spikes are deleted after firing/consuming


def configure(mode):
    Config.MODE = mode
    if mode == "quantized":
        Config.BLOCK_SHAPE = 2 #the size of the window block for the second layer
        Config.QUANTIZATION = True
        Config.Q_RANGE = 4
        Config.INVERT = False

        Config.TRAIN_SIZE = 1000
        Config.TEST_SIZE = 1000

        Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2) #784
        Config.NEURONS_LAYER2 = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) #49
        Config.NEURONS_LAYER1_2 = int(Config.NEURONS_LAYER1 + Config.NEURONS_LAYER2) #833
        Config.NEURONS_LAYER3 =  Config.NEURONS_LAYER1_2 + Config.CLASSES # aggiungo un layer con 8 neuroni 833+8=841
        Config.NEURONS_TOTAL = Config.NEURONS_LAYER3 + Config.CLASSES # 841

        Config.COMPARISON_THRESHOLD = 3  # threshold for the comparison during the quantized images processing


        Config.CSV_NAME = "SNPS_quantize.csv"
        Config.CSV_NAME_PRUNED = "SNPS_quantize_pruned.csv"

    if mode == "binarized":
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
        Config.IMG_SHAPE = 28
        Config.QUANTIZATION = True
        Config.Q_RANGE = 6
        Config.TRAIN_SIZE = 5
        Config.CLASSES = 8
        Config.INVERT = False

        Config.KERNEL_SHAPE = 3
        Config.KERNEL_NUMBER = 5

        Config.SEGMENTED_SHAPE = Config.IMG_SHAPE - Config.KERNEL_SHAPE + 1 # 26 or 6
        Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2) #784 or 64

        Config.CSV_NAME = "SNPS_cnn.csv"

    if mode == "flower":
        Config.MODE = "quantized"
        Config.TRAIN_SIZE = 500
        Config.TEST_SIZE = 500
        Config.IMG_SHAPE = 224
        Config.DATABASE = 'flower102'
        Config.CLASSES = 102

        Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2)
        Config.NEURONS_LAYER2 = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) #49
        Config.NEURONS_LAYER1_2 = int(Config.NEURONS_LAYER1 + Config.NEURONS_LAYER2) #833
        Config.NEURONS_TOTAL = Config.NEURONS_LAYER1_2 + Config.CLASSES # 841

    if mode == "digit":
        Config.WHITE_HOLE= True
        Config.Q_RANGE = 4
        Config.MODE = "cnn"
        Config.TRAIN_SIZE = 3
        Config.TEST_SIZE = 5
        Config.IMG_SHAPE = 28
        Config.BLOCK_SHAPE = 2
        Config.CLASSES = 10
        Config.INVERT = False

        Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2)
        Config.NEURONS_LAYER2 = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) #49
        Config.NEURONS_LAYER1_2 = int(Config.NEURONS_LAYER1 + Config.NEURONS_LAYER2) #833
        Config.NEURONS_LAYER3 =  Config.NEURONS_LAYER1_2 + Config.CLASSES # aggiungo un layer con 8 neuroni 833+8=841
        Config.NEURONS_TOTAL = Config.NEURONS_LAYER3 + Config.CLASSES # 841
