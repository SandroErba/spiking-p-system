from sps.SNPSystem import SNPSystem
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO
from skimage.measure import block_reduce

def compute_divisible_3():
    """SNPS that classify if a number is divisible by 3
    see Example 9 of paper https://link.springer.com/article/10.1007/s41965-020-00050-2?fromPaywallRec=false """
    snps = SNPSystem(5, 100, 'binary_spike_train')
    snps.load_neurons_from_csv("neuronsDiv3.csv")
    snps.spike_train = [1, 0, 0, 0, 0, 0, 0, 1]  # example of an input spike train that create halting computation
    snps.start()
    print(snps.history)


def compute_gen_even():
    """SNPS that generate all possible even numbers
    see Figure 3 of paper https://www.researchgate.net/publication/220443792_Spiking_Neural_P_Systems """
    snps = SNPSystem(5, 100, 'generative')
    snps.load_neurons_from_csv("neuronsGenerateEven.csv")
    snps.start()
    print(snps.history)


def compute_mnist():
    """example SNPS for mnist dataset"""
    # importa MNIST, rendilo 8x8, collega ogni pixel a un neurone, e questi neuroni a il secondo strato.
    # nel secondo strato ci sono 2 neuroni per 2 possibili output di esempio. ognuno dirà se è probabile o no che sia classificato così
    # neuron 64 classifica 1 come positivo, neurone 65 come negativo. neurone 66 classifica 8 come positivo, neurone 67 come negativo.
    mnist = fetch_openml('mnist_784', version=1, as_frame=False) # load MNIST
    X, y = mnist.data[:10], mnist.target[:10].astype(int)
    binary_8x8 = np.array([binarize_mnist_image(img) for img in X])
    flattened_images = binary_8x8.reshape((binary_8x8.shape[0], -1))
    snps = SNPSystem(5, 100, 'image_spike_train')
    snps.load_neurons_from_csv("neuronsMNIST.csv")
    snps.spike_train = flattened_images
    snps.start()
    print(snps.history)

    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)  # 2 righe, 5 colonne
        plt.imshow(binary_8x8[i], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f"Label: {y[i]}")
    plt.tight_layout()
    plt.show()

def binarize_mnist_image(img_flat, target_size=(8, 8), threshold=128):
    img = img_flat.reshape(28, 28)
    img_cropped = img[2:26, 2:26]  # Took central image
    img_resized = block_reduce(img_cropped, block_size=(3, 3), func=np.mean) #From 28x28 to 8x8
    img_resized = img_resized * 255
    binary_img = (img_resized > threshold).astype(int)
    return binary_img


def compute_blood_mnist():
    '''See BloodMNIST at https://medmnist.com/ '''
    input_number = 1
    info = INFO['bloodmnist']
    DataClass = getattr(medmnist, info['python_class'])
    dataset = DataClass(split='train', download=True)
    imgs, labels = dataset.imgs[:input_number], dataset.labels[:input_number].flatten()

    spike_train = []
    for img in imgs:
        ch_r, ch_g, ch_b = binarize_rgb_image(img)
        spike_train.extend([ch_r, ch_g, ch_b])

    spike_train = np.array(spike_train)  # shape (3 * input_number, 784)
    show_rgb_from_spike_train(spike_train, labels) # show the bynarized images
    snps = SNPSystem(5, 3, 'image_spike_train') #TODO attenzione ai max step
    snps.load_neurons_from_csv("neurons784image.csv")
    snps.spike_train = spike_train

    snps.start() #TODO try the code
    print(snps.history)

    plt.figure(figsize=(input_number, 4))
    for i in range(input_number):
        plt.subplot(2, 5, i + 1)  # TODO fixed value here
        plt.imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f"Label: {labels[i]}")
    plt.tight_layout()
    plt.show()

def binarize_rgb_image(img_rgb, threshold=128):
    # From [0,255] to [0,1]
    binary_channels = (img_rgb > threshold).astype(int)
    downsampled = []
    for c in range(3):
        ch = binary_channels[:, :, c]
        #ch_flat = ch.flatten().astype(int)  # 784 boolean
        downsampled.append(ch) #or ch_flat
    return downsampled  # List of 3 arrays of 784 bit

def show_rgb_from_spike_train(spike_train, labels):
    # Show the binarized images obtained
    num_images = len(labels)
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        r = spike_train[i * 3 + 0].reshape(28, 28)
        g = spike_train[i * 3 + 1].reshape(28, 28)
        b = spike_train[i * 3 + 2].reshape(28, 28)
        img_rgb = np.stack([r, g, b], axis=-1) * 255
        img_rgb = img_rgb.astype(np.uint8)
        plt.subplot(2, 5, i + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Label: {labels[i]}")
    plt.tight_layout()
    plt.show()