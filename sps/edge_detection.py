import numpy as np
from matplotlib import pyplot as plt

from sps.handle_csv import kernel_SNPS_csv
from sps.snp_system import SNPSystem
from sps import med_mnist
from sps.config import Config

def launch_gray_SNPS():
    (train_red, train_green, train_blue, train_labels), (_) = med_mnist.get_mnist_data('bloodmnist')
    kernel_SNPS_csv()

    final_results = []
    
    for i in range(Config.TRAIN_SIZE):
        # FIX QUI: .flatten() è fondamentale per passare da 28x28 a 784 neuroni
        single_image = train_red[i].flatten() 
        
        input_buffer = np.zeros((5, len(single_image)))
        input_buffer[0] = single_image

        snps = SNPSystem(5, 5, "images", "images", True)
        snps.load_neurons_from_csv("csv/" + Config.CSV_KERNEL_NAME)
        snps.spike_train = input_buffer
        snps.start()
        
        final_results.append(np.max(snps.edge_output, axis=1))

    show_images(np.column_stack(final_results))

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



