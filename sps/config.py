class Config:
    MODE = "cnn"
    DATABASE = 'digit'

    # IMAGES
    IMG_SHAPE = 28 #ipotizing squared shape images of 28 pixels
    BLOCK_SHAPE = 2  #the size of the window block for the second layer

    #SNPS BEHAVIOUR
    INVERT = False #invert or not invert the spike in the starting images #TODO ___tunable___?
    QUANTIZATION = True
    WHITE_HOLE= True #if true all the internal spikes are deleted after firing/consuming

    TRAIN_SIZE = 1000
    TEST_SIZE = 500

    #L1 - INPUT IMAGE
    NEURONS_L1 = int(IMG_SHAPE ** 2) #number of neurons for layer 1 (pixels in the image)
    Q_RANGE = 5 # the range of quantization, it works on images, rules and tuning #TODO ___tunable___

    #L2 - FEATURE EXTRACTION
    KERNELS = [
        [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
        [[0, 1, 1], [-1, 0, 1], [-1, -1, 0]],
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]],
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        [[0, -1, -1], [1, 0, -1], [1, 1, 0]],
        [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
        [[1, 1, 0], [1, 0, -1], [0, -1, -1]]
    ] #TODO ___tunable___, but with random ones is worse

    """KERNELS = [
        [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
        [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
        [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
    ] #TODO only 4 main directions"""

    KERNEL_NUMBER = len(KERNELS) #number of kernels in layer 2
    KERNEL_SHAPE = len(KERNELS[0]) #size of a single kernel
    SHAPE_FEATURE = int(IMG_SHAPE - KERNEL_SHAPE + 1) #size of a generated image in layer 2 (26)
    NEURONS_FEATURE = int(SHAPE_FEATURE ** 2) #number of neurons for each image in layer 2 (676)
    NEURONS_L2 = NEURONS_FEATURE * KERNEL_NUMBER #number of neurons for layer 2 (5408)

    #L3 - AVERAGE POOLING
    POOLING_SIZE = 2 #size of the pooling window
    SHAPE_POOL = int(SHAPE_FEATURE / POOLING_SIZE) #size of the resulting image after the pooling (13)
    NEURONS_POOL = int(SHAPE_POOL ** 2) #number of neurons for each image in layer 3 (169)
    NEURONS_L3 = int(KERNEL_NUMBER * NEURONS_POOL) #number of neurons on the pooling layer (1352)

    #L3 - CLASSIFICATION
    CLASSES = 10

    CSV_NAME = "SNPS_cnn.csv"

    SVM_C = 1.0 #TODO ___tunable___

    #MATRIX QUANTIZE #TODO ___tunable___
    QUANTIZE_METHOD = 3 #TODO add in GUI
    M_SPARSITY = 0.5 #percentage of 0 values in the quantized matrix, used if QUANTIZE_METHOD == 1
    M_POSITIVE = 0.25 #percentage of 1 values in the quantized matrix, used if QUANTIZE_METHOD == 1
    M_THRESHOLD = 0.5 #multiplied factor for column values, used if QUANTIZE_METHOD == 2

    #IMPORTANCE #TODO ___tunable___
    ALPHA_METHOD = 2 #how the model calculate the magnitude of the weights #TODO add in GUI
    DISCRETIZE_METHOD = 1 #how the model apply the *3 to rules #TODO add in GUI
    DISC_RANGE = 2

    # ENERGY COSTS
    WORST_REGEX = 100
    EXPECTED_REGEX = 10
    EXPECTED_SPIKE = 0.5

    THRESHOLD = 128 # higher Thr -> more spike


    @classmethod
    def compute_k_range(cls):
        cls.K_RANGE = [
            (
                sum(v == -1 for row in kernel for v in row) * -cls.Q_RANGE,
                sum(v == 1  for row in kernel for v in row) *  cls.Q_RANGE
            )
            for kernel in cls.KERNELS
        ]


def database(database):
    Config.DATABASE = database
    if database == "flower":
        Config.IMG_SHAPE = 224
        Config.CLASSES = 102


    if database == "digit":
        Config.IMG_SHAPE = 28
        Config.CLASSES = 10