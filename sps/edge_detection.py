import numpy as np
from matplotlib import pyplot as plt

from sps.handle_csv import kernel_SNPS_csv
from sps.med_image import show_images
from sps.snp_system import SNPSystem
from sps import med_mnist
from sps.config import Config

def launch_gray_SNPS():
    (train_red, train_green, train_blue, train_labels), (_) = med_mnist.get_mnist_data('bloodmnist')
    kernel_SNPS_csv()
    snps = SNPSystem(5, Config.TRAIN_SIZE + 5, "images", "images", True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)
    snps.spike_train = train_red
    snps.start()

    show_images(snps.image_output)





