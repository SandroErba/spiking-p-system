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
    SHAPE_FEATURE = IMG_SHAPE - KERNEL_SHAPE + 1
    NEURONS_LP = 0

    # STRING
    CSV_NAME = "Null"
    CSV_NAME_PRUNED = "Null"

    WHITE_HOLE= True #if true all the internal spikes are deleted after firing/consuming
    KERNELS = None


def configure(mode):
    Config.MODE = mode
    if mode == "quantized":
        Config.BLOCK_SHAPE = 2 #the size of the window block for the second layer
        Config.QUANTIZATION = True
        Config.Q_RANGE = 4
        Config.INVERT = False

        Config.TRAIN_SIZE = 1000
        Config.TEST_SIZE = 1000

        Config.NEURONS_L1 = int(Config.IMG_SHAPE ** 2) #784
        Config.NEURONS_L2 = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) #49
        Config.NEURONS_L12 = int(Config.NEURONS_L1 + Config.NEURONS_L2) #833
        Config.NEURONS_L3 = Config.NEURONS_L12 + Config.CLASSES # aggiungo un layer con 8 neuroni 833+8=841
        Config.NEURONS_T = Config.NEURONS_L3 + Config.CLASSES # 841

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

        Config.NEURONS_L1 = int(Config.IMG_SHAPE ** 2) #784
        Config.NEURONS_L2 = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) #49
        Config.NEURONS_L12 = int(Config.NEURONS_L1 + Config.NEURONS_L2) #833
        Config.NEURONS_T = Config.NEURONS_L12 + Config.CLASSES # 841

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
        Config.SHAPE_FEATURE = Config.IMG_SHAPE - Config.KERNEL_SHAPE + 1 # 27

        Config.NEURONS_L1 = int(Config.IMG_SHAPE ** 2) #784
        Config.NEURONS_L2 = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) #49
        Config.NEURONS_L12 = int(Config.NEURONS_L1 + Config.NEURONS_L2) #833
        Config.NEURONS_T = Config.NEURONS_L12 + Config.CLASSES # 841

    if mode == "cnn":
        Config.QUANTIZATION = True
        Config.TRAIN_SIZE = 5
        Config.TEST_SIZE = 5
        Config.Q_RANGE = 16
        Config.CSV_NAME = "SNPS_cnn.csv"

        Config.KERNELS = [
            [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
            [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]],
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
            [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]],
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
            [[0, -1, -1], [1, 0, -1], [1, 1, 0]],
            [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
            [[1, 1, 0], [1, 0, -1], [0, -1, -1]]
        ]

        Config.KERNEL_NUMBER = len(Config.KERNELS) #number of kernels in layer 2
        Config.KERNEL_SHAPE = len(Config.KERNELS[0]) #size of a single kernel
        Config.K_RANGE = [
            (
                sum(v == -1 for row in kernel for v in row) * -Config.Q_RANGE,
                sum(v == 1 for row in kernel for v in row) *  Config.Q_RANGE
            )
            for kernel in Config.KERNELS
        ]

        Config.NEURONS_L1 = int(Config.IMG_SHAPE ** 2) #number of neurons for layer 1 (pixels in the image)

        # Feature extraction
        Config.SHAPE_FEATURE = int(Config.IMG_SHAPE - Config.KERNEL_SHAPE + 1) #size of a generated image in layer 2 (26)
        Config.NEURONS_FEATURE = int(Config.SHAPE_FEATURE ** 2) #number of neurons for each image in layer 2 (676)
        Config.NEURONS_L2 = Config.NEURONS_FEATURE * Config.KERNEL_NUMBER #number of neurons for layer 2 (5408)
        Config.NEURONS_L12 = int(Config.NEURONS_L1 + Config.NEURONS_L2) #delete this

        # Average pooling
        Config.POOLING_SIZE = 2 #size of the pooling window


        Config.SHAPE_POOL = int(Config.SHAPE_FEATURE / Config.POOLING_SIZE) #size of the resulting image after the pooling (13)
        Config.NEURONS_POOL = int(Config.SHAPE_POOL ** 2) #number of neurons for each image in layer 3 (169)
        Config.NEURONS_LP = int(Config.KERNEL_NUMBER * Config.SHAPE_POOL ** 2) #number of neurons on the pooling layer (1352)

        #TODO NOW add an average pooling after layer 2 by simply /4 the actual charge. then show the resulting images

        Config.NEURONS_T = Config.NEURONS_L1 + Config.NEURONS_L2 + Config.NEURONS_LP #total number of neurons


def database(database):
    Config.DATABASE = database
    if database == "flower":
        Config.WHITE_HOLE= True
        Config.IMG_SHAPE = 224
        Config.CLASSES = 102
        Config.INVERT = False
        Config.BLOCK_SHAPE = 2


    if database == "digit":
        Config.WHITE_HOLE= True
        Config.IMG_SHAPE = 28
        Config.CLASSES = 10
        Config.INVERT = False
        Config.BLOCK_SHAPE = 2
