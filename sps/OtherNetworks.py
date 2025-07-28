from sps import Config
from sps.MedMnist import process_dataset
from sps.SNPSystem import SNPSystem
import numpy as np
from sklearn.datasets import fetch_openml, load_digits
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

# for testing different types of networks, or with different input shapes

def get_digits_data(): #8 x 8 database and relative model
    digits = load_digits()
    imgs = digits.images  # shape: (1797, 8, 8), grayscale
    labels = digits.target  # shape: (1797,)

    def to_rgb(img_gray):
        return np.stack([img_gray]*3, axis=-1).astype(np.uint8)

    rgb_imgs = [to_rgb(img) for img in imgs]

    class DummyDataset:
        def __init__(self, imgs, labels):
            self.imgs = imgs
            self.labels = labels

    full_dataset = DummyDataset(rgb_imgs, labels)
    train_dataset = DummyDataset(full_dataset.imgs[:Config.TRAIN_SIZE], full_dataset.labels[:Config.TRAIN_SIZE])
    test_dataset = DummyDataset(full_dataset.imgs[-Config.TEST_SIZE:], full_dataset.labels[-Config.TEST_SIZE:])
    return (
        process_dataset(train_dataset, Config.TRAIN_SIZE),
        process_dataset(test_dataset, Config.TEST_SIZE)
    )

def compute_divisible_3():
    #SNPS that classify if a number is divisible by 3
    #see Example 9 of paper https://link.springer.com/article/10.1007/s41965-020-00050-2?fromPaywallRec=false
    snps = SNPSystem(5, 100, 'binary_spike_train')
    snps.load_neurons_from_csv("neuronsDiv3.csv")
    snps.spike_train = [1, 0, 0, 0, 0, 0, 0, 1]  # example of an input spike train that create halting computation
    snps.start()
    print(snps.history)
    with open("historyDiv3.txt", "w", encoding="utf-8") as f:
        f.write(str(snps.history))

def compute_gen_even():
    #SNPS that generate all possible even numbers
    #see Figure 3 of paper https://www.researchgate.net/publication/220443792_Spiking_Neural_P_Systems
    #require nondeterminism, see method tick in class PNeuron
    snps = SNPSystem(5, 100, 'generative')
    snps.load_neurons_from_csv("neuronsGenerateEven.csv")
    snps.start()
    print(snps.history)
    
def compute_mnist():
    #example SNPS for mnist dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False) # load MNIST
    X, y = mnist.data[:10], mnist.target[:10].astype(int)
    binary_8x8 = np.array([binarize_mnist_image(img) for img in X])
    flattened_images = binary_8x8.reshape((binary_8x8.shape[0], -1))
    snps = SNPSystem(5, 100, Config.INPUT_TYPE)
    snps.load_neurons_from_csv("neuronsMNIST.csv")
    snps.spike_train = flattened_images
    snps.start()
    print(snps.history)

def binarize_mnist_image(img_flat, target_size=(8, 8), threshold=128):
    img = img_flat.reshape(28, 28)
    img_cropped = img[2:26, 2:26]  # Took central image
    img_resized = block_reduce(img_cropped, block_size=(3, 3), func=np.mean) #From 28x28 to 8x8
    img_resized = img_resized * 255
    binary_img = (img_resized > threshold).astype(int)
    return binary_img