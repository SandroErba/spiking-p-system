import numpy as np
import csv
from sps import Config
from sps.HandleCSV import binarized_SNPS_csv, quantized_SNPS_csv, prune_SNPS
from sps.HandleImage import get_blood_mnist_data
from sps.SNPSystem import SNPSystem

energy_tracker = {
    "worst": 0,  # worst case of energy spent
    "expected": 0    # expected case of energy spent
}

def update_energy(w_energy, e_energy):
    energy_tracker["worst"] += w_energy
    energy_tracker["expected"] += e_energy

def launch_binarized_SNPS():
    """Manage all the binarized SN P systems"""

    # Load and split database
    (train_red, train_green, train_blue, train_labels), \
        (test_red, test_green, test_blue, test_labels) = get_blood_mnist_data()

    # Group color channels
    train_channels = [train_red, train_green, train_blue]
    test_channels = [test_red, test_green, test_blue]

    predictions = []

    for train_data, test_data in zip(train_channels, test_channels):

        binarized_SNPS_csv()                    # prepare CSV for this channel
        rules_train_SNPS(train_data)            # adapt firing rules (layer 2)
        syn_train_SNPS(train_data, train_labels)  # prune + inhibit
        pred = compute_SNPS(test_data)          # test P system
        predictions.append(pred)

    # Unpack predictions
    red_pred, green_pred, blue_pred = predictions

    combined_ranking_score(red_pred, green_pred, blue_pred, test_labels)

    #print(f"Worst energy spent: {energy_tracker['worst']} fJ")
    #print(f"Expected energy spent: {energy_tracker['expected']} fJ")


def launch_quantized_SNPS():
    """Manage all quantized SN P systems"""
     #TODO something strange is happening, the number of step change at every color
    # Load and split database
    (train_red, train_green, train_blue, train_labels), \
        (test_red, test_green, test_blue, test_labels) = get_blood_mnist_data()

    # Group data into color channels
    train_channels = [train_red, train_green, train_blue]
    test_channels = [test_red, test_green, test_blue]

    predictions = []

    for train_data, test_data in zip(train_channels, test_channels):

        quantized_SNPS_csv()                # prepare CSV for this color
        syn_train_SNPS(train_data, train_labels)   # prune + inhibit
        pred = compute_SNPS(test_data)             # test
        predictions.append(pred)

    # Unpack predictions after loop
    red_pred, green_pred, blue_pred = predictions

    combined_ranking_score(red_pred, green_pred, blue_pred, test_labels)

    #print(f"Worst energy spent: {energy_tracker['worst']} fJ")
    #print(f"Expected energy spent: {energy_tracker['expected']} fJ")


def rules_train_SNPS(spike_train):
    snps = SNPSystem(5, Config.TRAIN_SIZE + 5, "images", "prediction", True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME_B)
    snps.spike_train = spike_train
    snps.layer_2_firing_counts = np.zeros(Config.NEURONS_LAYER2, dtype=int)
    w, e = snps.start()
    update_energy(w, e)

    normalize_rules(snps.layer_2_firing_counts.reshape((int(Config.IMG_SHAPE/Config.BLOCK_SHAPE), int(Config.IMG_SHAPE/Config.BLOCK_SHAPE))), Config.TRAIN_SIZE)

def syn_train_SNPS(spike_train, labels):
    snps = SNPSystem(5, Config.TRAIN_SIZE + 5, "images", "prediction", True)
    csv_name = Config.CSV_NAME_Q if Config.QUANTIZATION else Config.CSV_NAME_B
    snps.load_neurons_from_csv("csv/" + csv_name)
    snps.spike_train = spike_train
    snps.layer_2_synapses = np.zeros((Config.CLASSES, Config.NEURONS_LAYER2), dtype=float) # matrix for train synapses
    snps.labels = labels
    snps.layer_2_firing_counts = np.zeros(Config.NEURONS_LAYER2, dtype=int)
    w, e = snps.start() # run the SNPS
    update_energy(w, e)

    pruned_matrix = prune_matrix(snps.layer_2_synapses)
    prune_SNPS(pruned_matrix)

def compute_SNPS(spike_train):
    snps = SNPSystem(5, Config.TEST_SIZE + 5, "images", "prediction", True)
    csv_name_pruned = Config.CSV_NAME_Q_PRUNED if Config.QUANTIZATION else Config.CSV_NAME_B_PRUNED
    snps.load_neurons_from_csv("csv/" + csv_name_pruned)
    snps.spike_train = spike_train
    snps.layer_2_firing_counts = np.zeros(Config.NEURONS_LAYER2, dtype=int)
    w, e = snps.start() # run the SNPS
    update_energy(w, e)

    return snps.output_array[3:-2] #for merging the 3 results

def normalize_rules(firing_counts, imgs_number):
    """used in the rules training phase"""
    min_threshold = 1
    max_threshold = Config.BLOCK_SHAPE**2
    norm = firing_counts / imgs_number
    threshold_matrix = norm * (max_threshold - min_threshold) + min_threshold
    threshold_matrix = np.round(threshold_matrix).astype(int)
    binarized_SNPS_csv(threshold_matrix)

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


    # 1) Compute individual channel accuracies
    red_top1, red_top3 = evaluate_single_channel(pred_red, labels)
    green_top1, green_top3 = evaluate_single_channel(pred_green, labels)
    blue_top1, blue_top3 = evaluate_single_channel(pred_blue, labels)

    print("\n=== Individual Channel Accuracies ===")
    print(f"Red   - Top-1: {red_top1*100:.2f}%, Top-3: {red_top3*100:.2f}%")
    print(f"Green - Top-1: {green_top1*100:.2f}%, Top-3: {green_top3*100:.2f}%")
    print(f"Blue  - Top-1: {blue_top1*100:.2f}%, Top-3: {blue_top3*100:.2f}%")


    # 2) Existing combined ranking logic (unchanged)
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



