from sps.SNPSystem import SNPSystem
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

from skimage.measure import block_reduce


def compute_divisible_3():
    """SNPS that classify if a number is divisible by 3
    see Example 9 of paper https://link.springer.com/article/10.1007/s41965-020-00050-2?fromPaywallRec=false """
    snps = SNPSystem(5, 100, 'binary_spike_train')
    snps.load_neurons_from_csv("neuronsDiv3.csv")
    snps.spike_train = [1, 0, 0, 0, 0, 0, 0, 1]  # example of an input spike train that create halting computation
    snps.start()
    print(snps.history)
    with open("historyDiv3.txt", "w", encoding="utf-8") as f:
        f.write(str(snps.history))

def compute_gen_even():
    """SNPS that generate all possible even numbers
    see Figure 3 of paper https://www.researchgate.net/publication/220443792_Spiking_Neural_P_Systems """
    snps = SNPSystem(5, 100, 'generative')
    snps.load_neurons_from_csv("neuronsGenerateEven.csv")
    snps.start()
    print(snps.history)

def compute_mnist():
    """example SNPS for mnist dataset"""
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