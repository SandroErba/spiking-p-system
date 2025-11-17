import numpy as np
import medmnist
from medmnist import INFO
import csv
from sps import Config
from sps.SNPSystem import SNPSystem

energy_tracker = {
    "worst": 0,  # worst case of energy spent
    "expected": 0    # expected case of energy spent
}

def update_energy(w_energy, e_energy):
    energy_tracker["worst"] += w_energy
    energy_tracker["expected"] += e_energy

def get_blood_mnist_data(): # download the database
    info = INFO['bloodmnist']
    data_class = getattr(medmnist, info['python_class'])
    train_dataset = data_class(split='train', download=True)
    test_dataset = data_class(split='test', download=True)
    return (
        (process_dataset(train_dataset, Config.TRAIN_SIZE)),
        (process_dataset(test_dataset, Config.TEST_SIZE))
    )

def process_dataset(dataset, count): # flatten and split among color channels
    imgs = dataset.imgs[:count]
    labels = dataset.labels[:count].flatten()
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

def binarize_rgb_image(img_rgb): # binarize for create the input array
    binary_channels = 1 - (img_rgb > int(Config.THRESHOLD)).astype(int) # From [0,255] to [0,1]
    downsampled = []
    for c in range(3):
        ch = binary_channels[:, :, c]
        downsampled.append(ch)
    return downsampled  # List of 3 arrays of 784 bit

def launch_SNPS():
    """main method of the class, manage all the SN P systems"""
    (train_red, train_green, train_blue, train_labels), (test_red, test_green, test_blue, test_labels) = get_blood_mnist_data() # prepare and split database

    SNPS_csv() # red phase
    rules_train_SNPS(train_red) # adapt the firing rules in layer 2
    syn_train_SNPS(train_red, train_labels) # prune and inhibit synapses
    red_pred = compute_SNPS(test_red) # test the obtained P system

    SNPS_csv() # repeat for the other two color channels - green phase
    rules_train_SNPS(train_green)
    syn_train_SNPS(train_green, train_labels)
    green_pred = compute_SNPS(test_green)

    SNPS_csv() # blue phase
    rules_train_SNPS(train_blue)
    syn_train_SNPS(train_blue, train_labels)
    blue_pred = compute_SNPS(test_blue)

    combined_ranking_score(red_pred, green_pred, blue_pred, test_labels) # merge the three results

    print(f"Worst energy spent: {energy_tracker['worst']} fJ")
    print(f"Expected energy spent: {energy_tracker['expected']} fJ")

def rules_train_SNPS(spike_train):
    snps = SNPSystem(5, Config.TRAIN_SIZE + 5, "images", "prediction", True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)
    snps.spike_train = spike_train
    snps.layer_2_firing_counts = np.zeros(Config.NEURONS_LAYER2, dtype=int)
    w, e = snps.start()
    update_energy(w, e)

    normalize_rules(snps.layer_2_firing_counts.reshape((int(Config.IMG_SHAPE/Config.BLOCK_SHAPE), int(Config.IMG_SHAPE/Config.BLOCK_SHAPE))), Config.TRAIN_SIZE)

def syn_train_SNPS(spike_train, labels):
    snps = SNPSystem(5, Config.TRAIN_SIZE + 5, "images", "prediction", True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)
    snps.spike_train = spike_train
    snps.layer_2_synapses = np.zeros((Config.CLASSES, Config.NEURONS_LAYER2), dtype=float) # matrix for destroy synapses
    snps.labels = labels
    snps.layer_2_firing_counts = np.zeros(Config.NEURONS_LAYER2, dtype=int)
    w, e = snps.start()
    update_energy(w, e)

    pruned_matrix = prune_matrix(snps.layer_2_synapses)
    prune_SNPS(pruned_matrix)

def compute_SNPS(spike_train):
    snps = SNPSystem(5, Config.TEST_SIZE + 5, "images", "prediction", True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME_PRUNED)
    snps.spike_train = spike_train
    snps.layer_2_firing_counts = np.zeros(Config.NEURONS_LAYER2, dtype=int)
    w, e = snps.start()
    update_energy(w, e)

    return snps.output_array[3:-2] #for merging the 3 results

def normalize_rules(firing_counts, imgs_number):
    """used in the rules training phase"""
    min_threshold = 1
    max_threshold = Config.BLOCK_SHAPE**2
    norm = firing_counts / imgs_number
    threshold_matrix = norm * (max_threshold - min_threshold) + min_threshold
    threshold_matrix = np.round(threshold_matrix).astype(int)
    SNPS_csv(threshold_matrix)

def prune_matrix(synapses):
    """prepare the matrix for the synapses"""
    keep_matrix = np.zeros_like(synapses, dtype=int)  # start with all 0
    for class_idx in range(synapses.shape[0]):
        weights = synapses[class_idx]
        num_excite = int((1 - Config.PRUNING_PERC - Config.INHIBIT_PERC) * len(weights))
        num_inhibit = int(Config.INHIBIT_PERC * len(weights))

        excite_indices = np.argsort(weights)[-num_excite:]
        inhibit_indices = np.argsort(weights)[:num_inhibit]
        keep_matrix[class_idx, excite_indices] = 1
        keep_matrix[class_idx, inhibit_indices] = -1
    return keep_matrix

def prune_SNPS(pruned_matrix):
    """change the synapses in the csv file"""
    with open("csv/" + Config.CSV_NAME, 'r') as f_in, open("csv/" + Config.CSV_NAME_PRUNED, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            neuron_id = int(row[0])
            if Config.NEURONS_LAYER1 <= neuron_id < Config.NEURONS_LAYER1_2:
                neuron_index = neuron_id - Config.NEURONS_LAYER1
                pruned_outputs = []
                for class_idx in range(Config.CLASSES):
                    val = pruned_matrix[class_idx][neuron_index]
                    if val != 0:
                        target_id = Config.NEURONS_LAYER1_2 + class_idx
                        if val == -1:
                            target_id = -target_id  # inhibitory
                        pruned_outputs.append(str(target_id))
                row[2] = "[" + ", ".join(pruned_outputs) + "]"

            writer.writerow(row)

def combined_ranking_score(pred_red, pred_green, pred_blue, labels):
    """calculate the model's performance including per-channel and combined ranking"""

    def evaluate_single_channel(predictions, labels):
        """Return top-1 and top-3 accuracy for one color channel."""
        top1, top3 = 0, 0
        for row, true_label in zip(predictions, labels):
            noise = np.random.rand(Config.CLASSES) * 1e-6
            ranking = np.argsort(-(row + noise))
            rank = int(np.where(ranking == true_label)[0][0])
            if rank == 0:
                top1 += 1
            if rank < 3:
                top3 += 1
        n = len(labels)
        return top1 / n, top3 / n


    # ----------------------------------------------------------
    # 1) Compute individual channel accuracies
    # ----------------------------------------------------------
    red_top1, red_top3 = evaluate_single_channel(pred_red, labels)
    green_top1, green_top3 = evaluate_single_channel(pred_green, labels)
    blue_top1, blue_top3 = evaluate_single_channel(pred_blue, labels)

    print("\n=== Individual Channel Accuracies ===")
    print(f"Red   - Top-1: {red_top1*100:.2f}%, Top-3: {red_top3*100:.2f}%")
    print(f"Green - Top-1: {green_top1*100:.2f}%, Top-3: {green_top3*100:.2f}%")
    print(f"Blue  - Top-1: {blue_top1*100:.2f}%, Top-3: {blue_top3*100:.2f}%")


    # ----------------------------------------------------------
    # 2) Existing combined ranking logic (unchanged)
    # ----------------------------------------------------------
    scores = []
    top1_correct = 0
    top3_correct = 0

    per_class_correct = np.zeros(Config.CLASSES, dtype=int)
    class_counts = np.zeros(Config.CLASSES, dtype=int)

    for red_row, green_row, blue_row, true_label in zip(pred_red, pred_green, pred_blue, labels):
        noise = np.random.rand(Config.CLASSES) * 1e-6
        red_rank = np.argsort(-red_row - noise)
        green_rank = np.argsort(-green_row - noise)
        blue_rank = np.argsort(-blue_row - noise)

        combined_score = np.zeros(Config.CLASSES, dtype=float)
        for i in range(Config.CLASSES):
            combined_score[i] = (
                    np.where(red_rank == i)[0][0] +
                    np.where(green_rank == i)[0][0] +
                    np.where(blue_rank == i)[0][0]
            )

        final_ranking = np.argsort(combined_score)
        rank = int(np.where(final_ranking == true_label)[0][0])
        scores.append(rank)

        if rank < 3:
            top3_correct += 1
            if rank == 0:
                top1_correct += 1
                per_class_correct[true_label] += 1

        class_counts[true_label] += 1

    # Combined metrics
    top1_accuracy = top1_correct / len(labels)
    top3_accuracy = top3_correct / len(labels)
    avg_rank = sum(scores) / len(scores)

    print("\n=== Combined Ranking Accuracy ===")
    print("Mean score:", avg_rank)
    print("Top-1 accuracy:", round(top1_accuracy * 100, 2), "%")
    print("Top-3 accuracy:", round(top3_accuracy * 100, 2), "%")

    print("\nPer-class accuracy:")
    for i in range(Config.CLASSES):
        count = class_counts[i]
        correct = per_class_correct[i]
        acc = (correct / count) * 100 if count > 0 else 0
        print(f"  Class {i}: {acc:.2f}% accuracy over {count} instances")

    return (
        scores,
        avg_rank,
        top1_accuracy,
        top3_accuracy,
        (red_top1, red_top3),
        (green_top1, green_top3),
        (blue_top1, blue_top3)
    )


def SNPS_csv(threshold_matrix=None, filename="csv/" + Config.CSV_NAME):
    """Generate the SN P system to analize chosen images
    If a matrix is passed, update the existing P system"""
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # Layer 1: Input RGB (784 neurons) from 28x28 to 7x7 using 4x4 blocks
        for neuron_id in range(Config.NEURONS_LAYER1):
            block_row = (neuron_id // Config.IMG_SHAPE) // Config.BLOCK_SHAPE
            block_col = (neuron_id % Config.IMG_SHAPE) // Config.BLOCK_SHAPE
            block_id = block_row * int(Config.IMG_SHAPE/Config.BLOCK_SHAPE) + block_col
            output_neuron = Config.NEURONS_LAYER1 + block_id

            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                f"[{output_neuron}]", # output_targets
                0,                    # neuron_type
                "[0,1,1,1,0]"         # firing rule
            ])

        # Layer 2: Pooling (49 neurons) - id 784–832
        output_targets = str(list(range(Config.NEURONS_LAYER1_2, Config.NEURONS_TOTAL))) # for firing at the output neurons
        if threshold_matrix is None:
            for neuron_id in range(Config.NEURONS_LAYER1, Config.NEURONS_LAYER1_2):
                writer.writerow([
                    neuron_id,            # id
                    0,                    # initial_charge
                    output_targets,       # output_targets
                    1,                    # neuron_type
                    "[1,1,0,1,0]",       # firing rule if c >= 1
                    "[1,1,1,0,0]"        # forgetting rule if didn't fire
                ])

        else: # change the P system using the new charges for the firing rules
            threshold_array = threshold_matrix.flatten()
            for neuron_id in range(Config.NEURONS_LAYER1, Config.NEURONS_LAYER1_2):
                firing_threshold = threshold_array[neuron_id - Config.NEURONS_LAYER1]
                firing_rule = f"[1,{firing_threshold},0,1,0]"
                writer.writerow([
                    neuron_id,            # id
                    0,                    # initial_charge
                    output_targets, # output_targets
                    1,                    # neuron_type
                    firing_rule,          # firing rule based on input matrix
                    "[1,1,1,0,0]"        # forgetting rule if didn't fire
                ])

        # Layer 3: Output (8 neurons) - id 833–840
        for neuron_id in range(Config.NEURONS_LAYER1_2, Config.NEURONS_TOTAL):
            #label = neuron_id - Config.NEURONS_LAYER1_2
            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                "[]",                 # output_targets
                2,                    # neuron_type
                "[1,1,1,0,0]"         # forgetting rule
            ])
