import csv
import numpy as np
from matplotlib import pyplot as plt
from sps.SNPSystem import SNPSystem
from sps import MedMnist, Config


def launch_gray_SNPS():
    (train_red, train_green, train_blue, train_labels), (test_red, test_green, test_blue, test_labels) = MedMnist.get_blood_mnist_data()
    kernel_SNPS_csv()
    snps = SNPSystem(5, Config.TRAIN_SIZE + 5, "images", "images", True)
    snps.load_neurons_from_csv(Config.CSV_KERNEL_NAME)
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


def kernel_SNPS_csv():
    """
    Generate a 3-layer SN P system to perform edge detection on a 28x28 image
    using 6 convolution kernels (2x2) with values 1 and -1.
    The structure is:
    - Layer 1: Input neurons (784 neurons), firing to 6 parallel subnetworks
    - Layer 2: One 27x27 grid per kernel (6 kernels → 4374 neurons)
    - Layer 3: 27x27 neurons (729 neurons), sum of all filtered maps
    for more info see: Ultrafast neuromorphic photonic image processing with aVCSEL neuron"""
    kernels = [
        [[1, -1], [1, -1]], # Vertical 1
        [[-1, 1], [-1, 1]], # Vertical 2
        [[-1, -1], [1, 1]], # Horizontal 1
        [[1, 1], [-1, -1]], # Horizontal 2
        [[-1, 1], [1, -1]], # Diagonal 1
        [[1, -1], [-1, 1]]  # Diagonal 2
    ]
    layer1_size = Config.IMG_SHAPE * Config.IMG_SHAPE
    layer2_size_per_kernel = Config.SEGMENTED_SHAPE * Config.SEGMENTED_SHAPE
    total_layer2_size = layer2_size_per_kernel * len(kernels)
    layer3_offset = Config.NEURONS_LAYER1 + total_layer2_size

    with open(Config.CSV_KERNEL_NAME, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # Layer 1: Input a 28x28 grayscale image
        for neuron_id in range(layer1_size):
            i_row = neuron_id // Config.IMG_SHAPE
            i_col = neuron_id % Config.IMG_SHAPE
            output_targets = []

            for k_index, kernel in enumerate(kernels):
                layer2_offset = Config.NEURONS_LAYER1 + k_index * layer2_size_per_kernel

                for ki in range(Config.KERNEL_SHAPE):
                    for kj in range(Config.KERNEL_SHAPE):
                        o_row = i_row - ki
                        o_col = i_col - kj

                        if 0 <= o_row < Config.SEGMENTED_SHAPE and 0 <= o_col < Config.SEGMENTED_SHAPE:
                            output_idx = o_row * Config.SEGMENTED_SHAPE + o_col
                            target_id = layer2_offset + output_idx
                            weight = kernel[ki][kj]
                            if weight == 1:
                                output_targets.append(target_id)
                            elif weight == -1:
                                output_targets.append(-target_id)

            writer.writerow([
                neuron_id,                     # id
                0,                             # initial_charge
                str(output_targets),          # output_targets
                0,                             # neuron_type
                "[0,1,1,1,0]"                  # firing rule
            ])

        # Layer 2: Accumulate spikes from the kernels
        for k_index in range(len(kernels)):
            layer2_offset = Config.NEURONS_LAYER1 + k_index * layer2_size_per_kernel

            for i in range(layer2_size_per_kernel):
                output_target = layer3_offset + i  # Same i-th neuron in layer 3
                writer.writerow([
                    layer2_offset + i,       # id
                    0,                       # initial_charge
                    f"[{output_target}]",    # output_targets
                    1,                       # neuron_type
                    "[0,2,0,1,0]"            # Fires only if c == 2 because 2 is the kernel threshold value
                ])

        # Layer 3: Aggregation neurons
        for i in range(layer2_size_per_kernel):
            writer.writerow([
                layer3_offset + i,           # id
                0,                           # initial_charge
                "[]",                        # output_targets
                2,                           # neuron_type (output)
                "[1,1,0,0,0]"                # Forgetting rule
            ])
