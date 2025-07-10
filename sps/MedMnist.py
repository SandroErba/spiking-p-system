import numpy as np
import medmnist
import matplotlib.pyplot as plt
from medmnist import INFO
import csv

from sps import PNeuron
from sps.SNPSystem import SNPSystem


def get_spike_train_from_blood_mnist(input_number, skip_number):
    info = INFO['bloodmnist']
    DataClass = getattr(medmnist, info['python_class'])

    #TODO sistemare e testare questo codice, toglier skip number
    def process_dataset(dataset, skip, count):
        imgs = dataset.imgs[skip:skip+count]
        labels = dataset.labels[skip:skip+count].flatten()
        red_channel = []
        green_channel = []
        blue_channel = []

        for img in imgs:
            ch_r, ch_g, ch_b = binarize_rgb_image(img)
            red_channel.append(ch_r)
            green_channel.append(ch_g)
            blue_channel.append(ch_b)

        return (
            np.array(red_channel),
            np.array(green_channel),
            np.array(blue_channel),
            labels
        )

    train_dataset = DataClass(split='train', download=True) # Train
    train_red, train_green, train_blue, train_labels = process_dataset(train_dataset, skip_number, input_number)

    test_dataset = DataClass(split='test', download=True) # Test
    test_red, test_green, test_blue, test_labels = process_dataset(test_dataset, 0, len(test_dataset))

    return (
        (train_red, train_green, train_blue, train_labels),
        (test_red, test_green, test_blue, test_labels)
    )

def binarize_rgb_image(img_rgb, threshold=128):
    # From [0,255] to [0,1]
    binary_channels = 1 - (img_rgb > threshold).astype(int)
    downsampled = []
    for c in range(3):
        ch = binary_channels[:, :, c]
        #ch_flat = ch.flatten().astype(int)  # 784 boolean
        downsampled.append(ch) #or ch_flat
    return downsampled  # List of 3 arrays of 784 bit

def launch_blood(imgs_number, pruning_perc):
    spike_train_red, spike_train_rgb, labels = get_spike_train_from_blood_mnist(imgs_number, 0)
    blood_SNPsystem_csv()
    rules_train_blood_mnist(imgs_number, spike_train_red)
    syn_train_blood_mnist(imgs_number, spike_train_red, labels, pruning_perc)
    compute_blood_mnist(imgs_number, spike_train_red, labels)

def rules_train_blood_mnist(imgs_number, spike_train):
    snps = SNPSystem(5, imgs_number + 5, 'image_spike_train')
    snps.load_neurons_from_csv("neurons784image.csv")
    snps.spike_train = spike_train
    snps.layer_2_firing_counts = np.zeros(49, dtype=int)
    snps.start()

    normalize_rules(snps.layer_2_firing_counts.reshape((7, 7)), imgs_number)

def syn_train_blood_mnist(imgs_number, spike_train, labels, pruning_perc):
    snps = SNPSystem(5, imgs_number + 5, 'image_spike_train')
    snps.load_neurons_from_csv("neurons784image.csv")
    snps.spike_train = spike_train
    snps.layer_2_synapses = np.zeros((8, 49), dtype=float) # matrix for destroy synapses
    snps.labels = labels
    snps.layer_2_firing_counts = np.zeros(49, dtype=int)
    snps.start()

    pruned_matrix = prune_matrix(snps.layer_2_synapses, pruning_perc)
    prune_PSystem(pruned_matrix, "neurons784image.csv", "neurons784image_pruned.csv")


def compute_blood_mnist(imgs_number, spike_train, labels):
    snps = SNPSystem(5, imgs_number + 5, 'image_spike_train')
    snps.load_neurons_from_csv("neurons784image_pruned.csv")
    snps.spike_train = spike_train
    snps.layer_2_firing_counts = np.zeros(49, dtype=int)
    snps.start()

    print("array finale delle prediction: ", snps.output_array[3:-2])
    print("Array delle labels: ", labels)
    ranking_score(snps.output_array[3:-2], labels)


def normalize_rules(firing_counts, imgs_number):
    plt.imshow(firing_counts, cmap='hot')
    plt.title("Firing counts of layer 2")
    plt.colorbar()
    plt.show()
    print(firing_counts)

    min_threshold = 1
    max_threshold = 16

    norm = firing_counts / imgs_number
    threshold_matrix = norm * (max_threshold - min_threshold) + min_threshold
    threshold_matrix = np.round(threshold_matrix).astype(int)

    print(threshold_matrix)
    blood_SNPsystem_csv(threshold_matrix)


def prune_matrix(synapses, percentage = 0.5):
    keep_matrix = np.ones_like(synapses, dtype=int)  # 8x49 made of 1

    for class_idx in range(synapses.shape[0]):
        weights = synapses[class_idx]
        num_to_prune = int(percentage * len(weights))

        prune_indices = np.argsort(weights)[:num_to_prune] # Index to prun
        keep_matrix[class_idx, prune_indices] = 0

    print("Matrix for prune: ", keep_matrix)
    return keep_matrix

def prune_PSystem(pruned_matrix, filename="neurons784image.csv", output="neurons784image_pruned.csv"):
    with open(filename, 'r') as f_in, open(output, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        header = next(reader)
        writer.writerow(header)
        # TODO potrei eliminare neuroni con 0 output
        for row in reader:
            neuron_id = int(row[0])
            if 784 <= neuron_id <= 832:
                neuron_index = neuron_id - 784
                pruned_outputs = [
                    str(833 + class_idx)
                    for class_idx in range(8)
                    if pruned_matrix[class_idx][neuron_index] == 1
                ]
                row[2] = "[" + ", ".join(pruned_outputs) + "]" #TODO qui è dove metterei gli anti-spike

            writer.writerow(row)

def ranking_score(predictions, labels):
    scores = []
    top1_correct = 0
    top3_correct = 0

    for pred_row, true_label in zip(predictions, labels):
        sorted_indices = np.argsort(-pred_row)
        rank = int(np.where(sorted_indices == true_label)[0][0])
        scores.append(rank)

        if rank == 0:
            top1_correct += 1
        if rank < 3:
            top3_correct += 1

    top1_accuracy = top1_correct / len(labels)
    top3_accuracy = top3_correct / len(labels)
    avg_rank = sum(scores) / len(scores)

    print("Score:", scores)
    print("Mean score:", avg_rank)
    print("Top-1 accuracy:", round(top1_accuracy * 100, 2), "%")
    print("Top-3 accuracy:", round(top3_accuracy * 100, 2), "%")

    return scores, avg_rank, top1_accuracy, top3_accuracy

def blood_SNPsystem_csv(threshold_matrix=None, filename="neurons784image.csv"):
    """Generate the SN P system to analize blood mnist images
    If a matrix is passed, update the existing P system"""
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
        if threshold_matrix is None:
            for neuron_id in range(784, 784 + 49):
                writer.writerow([
                    neuron_id,            # id
                    0,                    # initial_charge
                    "[833, 834, 835, 836, 837, 838, 839, 840]", # output_targets
                    1,                    # neuron_type
                    "[-1,0,1,1,0]",       # firing rule if c >= 1
                    "[-1,0,1,0,0]"        # forgetting rule if didn't fire
                ])

        else: # change the P system using the new charges for the firing rules
            threshold_array = threshold_matrix.flatten()
            for neuron_id in range(784, 784 + 49):
                firing_threshold = threshold_array[neuron_id-784]
                firing_rule = f"[-1,0,{firing_threshold},1,0]"
                writer.writerow([
                    neuron_id,            # id
                    0,                    # initial_charge
                    "[833, 834, 835, 836, 837, 838, 839, 840]", # output_targets
                    1,                    # neuron_type
                    firing_rule,          # firing rule based on input matrix
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