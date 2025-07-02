import numpy as np
import medmnist
import matplotlib.pyplot as plt
from medmnist import INFO
import csv
from sps.SNPSystem import SNPSystem


def get_spike_train_from_blood_mnist(input_number, skip_number):
    info = INFO['bloodmnist']
    DataClass = getattr(medmnist, info['python_class'])
    dataset = DataClass(split='train', download=True)
    imgs, labels = dataset.imgs[skip_number:input_number+skip_number], dataset.labels[skip_number:input_number+skip_number].flatten()
    #show_images(imgs, labels)

    spike_train_red = []
    spike_train_rgb = []
    for img in imgs:
        ch_r, ch_g, ch_b = binarize_rgb_image(img)
        spike_train_rgb.extend([ch_r, ch_g, ch_b])
        spike_train_red.append(ch_r)  # Only red channel

    spike_train_red = np.array(spike_train_red)  # shape (input_number, 784)
    return spike_train_red, spike_train_rgb, labels

def binarize_rgb_image(img_rgb, threshold=128):
    # From [0,255] to [0,1]
    binary_channels = 1 - (img_rgb > threshold).astype(int)
    downsampled = []
    for c in range(3):
        ch = binary_channels[:, :, c]
        #ch_flat = ch.flatten().astype(int)  # 784 boolean
        downsampled.append(ch) #or ch_flat
    return downsampled  # List of 3 arrays of 784 bit


def compute_blood_mnist():
    spike_train_red, spike_train_rgb, labels = get_spike_train_from_blood_mnist(10, 0)
    #show_rgb_from_spike_train(spike_train_rgb, labels) # show the bynarized images
    #show_red_from_spike_train(spike_train_red, labels)

    snps = SNPSystem(5, 15, 'image_spike_train')
    snps.load_neurons_from_csv("neurons784image.csv")
    snps.spike_train = spike_train_red
    snps.start()
    with open("history784image.html", "w", encoding="utf-8") as f:
        f.write(f"<pre>{str(snps.history)}</pre>")


def rules_train_rgb_blood_mnist():
    spike_train_red, spike_train_rgb, labels = get_spike_train_from_blood_mnist(10, 0)

    snps = SNPSystem(5, 15, 'image_spike_train')
    snps.load_neurons_from_csv("neurons784image.csv")
    snps.spike_train = spike_train_red

    layer_2_firing_counts = np.zeros(49, dtype=int)
    snps.layer_2_firing_counts = layer_2_firing_counts

    snps.start()
    with open("history784image.html", "w", encoding="utf-8") as f:
        f.write(f"<pre>{str(snps.history)}</pre>")

    plt.imshow(snps.layer_2_firing_counts.reshape((7, 7)), cmap='hot')
    plt.title("Firing counts of layer 2")
    plt.colorbar()
    plt.show()
    print(snps.layer_2_firing_counts.reshape((7, 7)))






"""Code for showing full, binarized and red images"""
def show_images(imgs, labels):
    num_images = len(labels)
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        img_rgb = imgs[i]
        plt.subplot(2, 5, i + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Label: {labels[i]}")
    plt.tight_layout()
    plt.show()

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

def show_red_from_spike_train(spike_train, labels):
    num_images = len(labels)
    plt.figure(figsize=(10, 4))
    for i in range(num_images):
        r = spike_train[i].reshape(28, 28)
        g = np.zeros_like(r)
        b = np.zeros_like(r)
        img_rgb = np.stack([r, g, b], axis=-1) * 255
        img_rgb = img_rgb.astype(np.uint8)
        plt.subplot(2, 5, i + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Label: {labels[i]}")
    plt.tight_layout()
    plt.show()


def create_blood_network_csv(filename="neurons784image.csv"):
    """Generate the SN P system to analize blood mnist images"""
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # Layer 1: Input RGB (784 neurons) from 28x28 to 7x7 using 4x4 blocks
        for neuron_id in range(784):
            row = neuron_id // 28
            col = neuron_id % 28
            block_row = row // 4
            block_col = col // 4
            block_id = block_row * 7 + block_col
            output_neuron = 784 + block_id

            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                f"[{output_neuron}]", # output_targets
                0,                    # neuron_type
                "[0,1,1,1,0]"         # firing rule
            ])

        # Layer 2: Pooling (49 neurons) - id 784–832
        for neuron_id in range(784, 784 + 49):
            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                "[833, 834, 835, 836, 837, 838, 839, 840]", # output_targets
                1,                    # neuron_type
                "[-1,0,2,1,0]",       # firing rule if c >= 2
                "[-1,0,1,0,0]"        # forgetting rule if didn't fire
            ])

        # Layer 3: Output (8 neurons) - id 833–840
        for neuron_id in range(833, 841):
            label = neuron_id - 833
            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                "[]", # output_targets
                2,                    # neuron_type
                "[1,0,1,0,0]"         # forgetting rule
            ])

    print(f"Created {filename} with 841 neurons")