import csv
import numpy as np
from matplotlib import pyplot as plt

from sps.HandleCSV import kernel_SNPS_csv
from sps.SNPSystem import SNPSystem
from sps import MedMnist, Config


def launch_gray_SNPS():
    (train_red, train_green, train_blue, train_labels), (test_red, test_green, test_blue, test_labels) = MedMnist.get_blood_mnist_data()
    kernel_SNPS_csv()
    snps = SNPSystem(5, Config.TRAIN_SIZE + 5, "images", "images", True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_KERNEL_NAME)
    snps.spike_train = train_red
    snps.start()

    show_images(snps.edge_output)

def show_images(output_array, img_size=27, max_images=Config.TRAIN_SIZE):
    images = np.asarray(output_array)
    num_images = min(images.shape[1], max_images)
    cols = min(num_images, 5)
    rows = int(np.ceil(num_images / cols))
    plt.figure(figsize=(2.5 * cols, 2.5 * rows))
    for i in range(num_images):
        img = images[:, i].reshape((img_size, img_size))
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Image {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()



