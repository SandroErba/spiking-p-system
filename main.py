"""Some examples using spiking neural p systems"""
from skimage.measure import block_reduce

from sps.SNPSystem import SNPSystem
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

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
    # nel secondo strato ci sono 2 neuroni per ogni possibile output. ognuno dirà se è probabile o no che sia classificato così
    # neuron 64 classifica 1 come positivo, neurone 65 come negativo.
    # neurone 66 classifica 8 come positivo, neurone 67 come negativo.
    # TODO controllare output e generare regole di firing nei 4 neuroni output cche salvino i dati d iclassificazione

    mnist = fetch_openml('mnist_784', version=1, as_frame=False) # load MNIST
    X, y = mnist.data[:10], mnist.target[:10].astype(int)
    binary_8x8 = np.array([binarize_mnist_image(img) for img in X])
    flattened_images = binary_8x8.reshape((binary_8x8.shape[0], -1))
    print(flattened_images)
    snps = SNPSystem(5, 100, '8x8_spike_train')
    snps.load_neurons_from_csv("neuronsMNIST.csv")
    snps.spike_train = flattened_images
    snps.start()
    print(snps.history)

    # ---Visualizza le prime 10 immagini binarizzate 8x8
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


#compute_divisible_3()
compute_mnist()

'''How to read the output table:
r: rule applied. r:999a+2;3->0;1 mean "using 3 charge, 0 spike are fired, with 1 delay". ! = firing rules. 
1a is the condition part that follows: if c %999 == 2, the rule applies
c: internal charge of neuron. i:x(y) means "x charge received from neuron y" 

Remember: a neuron has rules of type E/r^x->a. E is a regular expression that should find an EXACT match with the number of spikes a in the neuron.
the rule consumes x spikes, or all the spikes if x is no defined. We use div and mod to describe E, and source is x. 
For the rules that want exact numbers, not regulars expressions, we are using div = k as "large number"
'''