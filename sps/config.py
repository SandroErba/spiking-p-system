class Config:
    MODE = "cnn"
    DATABASE = 'digit'

    # IMAGES
    IMG_SHAPE = 28 #ipotizing squared shape images of 28 pixels
    CLASSES = 10
    BLOCK_SHAPE = None  #the size of the window block for the second layer


    THRESHOLD = 128 # higher Thr -> more spike

    #SNPS BEHAVIOUR
    INVERT = False #invert or not invert the spike in the starting images (0...4) -> (4...0) #TODO ___trainable___?
    QUANTIZATION = True
    WHITE_HOLE= True #if true all the internal spikes are deleted after firing/consuming


    TRAIN_SIZE = 2000
    TEST_SIZE = TRAIN_SIZE



    #L1 - INPUT IMAGE
    NEURONS_L1 = int(IMG_SHAPE ** 2) #number of neurons for layer 1 (pixels in the image)
    Q_RANGE = 16 # the range of quantization, it works on images, rules and tuning #TODO ___trainable___

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
    ] #TODO ___trainable___, but with random ones is worse

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

    CSV_NAME = "SNPS_cnn.csv"

    #PERCEPTRON
    SPARSITY = 0.5 #TODO ___trainable___ percentage of 0 values in the perceptron
    POSITIVE = 0.25 #TODO ___trainable___ percentage of 1 values in the perceptron
    LR = 0.02 #TODO ___trainable___


    # ENERGY COSTS
    WORST_REGEX = 100
    EXPECTED_REGEX = 10
    EXPECTED_SPIKE = 0.5


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
        Config.BLOCK_SHAPE = 2


    if database == "digit":
        Config.IMG_SHAPE = 28
        Config.CLASSES = 10
        Config.BLOCK_SHAPE = 2
