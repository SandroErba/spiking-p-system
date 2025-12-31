import matplotlib.pyplot as plt
import numpy as np
import medmnist
from medmnist import INFO
import csv
from sps import config
from sps.config import Config
from sps.snp_system import SNPSystem
import os

# Define project root for absolute paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def visualize_batch(dataset, indices, thresholds):
    """
    Visualizes the images specified in the 'indices' list in a SINGLE multiple plot.
    Each image occupies 3 rows (R, G, B).
    """
    
    # Setup parameters
    interval_limits = [0] + sorted(thresholds) + [256]
    num_time_steps = len(interval_limits) - 1
    num_images = len(indices)
    
    # Calculate grid dimensions
    # Rows: Number of images * 3 channels (R, G, B)
    # Columns: 1 (Original) + Number of Time Steps
    n_rows = num_images * 3
    n_cols = 1 + num_time_steps
    
    # Dynamic figure height (3 inches per row)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    
    # General title
    #fig.suptitle(f"Temporal Analysis on {num_images} images (Thresholds: {thresholds})", fontsize=16)
    
    channel_names = ['R', 'G', 'B']
    
    # --- LOOP OVER IMAGES ---
    for i, img_idx in enumerate(indices):
        img = dataset.imgs[img_idx]
        label = dataset.labels[img_idx]
        
        # --- LOOP OVER CHANNELS (R, G, B) ---
        for c in range(3):
            # Calculate which row of the plot we are on
            # Ex: Image 0 -> rows 0,1,2. Image 1 -> rows 3,4,5...
            current_row = (i * 3) + c
            
            ch_data = img[:, :, c]
            
            # --- COLUMN 0: Original Channel Image ---
            ax_orig = axes[current_row, 0]
            ax_orig.imshow(ch_data, cmap='gray', vmin=0, vmax=255)
            
            # Side labels to understand what we are looking at
            if c == 1: # Only on the central row of the image (Green) put the ID
                ax_orig.set_ylabel(f"IMG {img_idx}\n(Class {label})", fontsize=14, fontweight='bold', labelpad=20)
            
            # Channel label
            ax_orig.text(-0.3, 0.5, channel_names[c], transform=ax_orig.transAxes, 
                         va='center', ha='right', fontsize=12, rotation=90)

            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            if current_row == 0: ax_orig.set_title("Original\nChannel", fontsize=12)

            # --- COLUMNS 1..N: Time Steps ---
            for t in range(num_time_steps):
                t_low = interval_limits[t]
                t_high = interval_limits[t+1]
                
                # Binary mask (Your SNPS logic)
                mask = np.logical_and(ch_data >= t_low, ch_data < t_high)
                
                ax_t = axes[current_row, t + 1]
                ax_t.imshow(mask, cmap='binary_r') # Black on white
                ax_t.axis('off')
                
                # Titles only on the very first row
                if current_row == 0:
                    ax_t.set_title(f"T={t+1}\nRange [{t_low}-{t_high})", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Leave space for the title above
    plt.show()



energy_tracker = {
    "worst": 0,  
    "expected": 0    
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


def temporal_encode_image(img_rgb, thresholds):
    """
    Encodes the image into a time-resolved input array (Latency Coding).
    Each binary map activates only for pixels in its specific intensity interval
    (non-cumulative), mapping darkness to time.
    """
    
    # Create intensity interval limits: [0, T1), [T1, T2), ..., [TN, 256]
    interval_limits = [0] + sorted(thresholds) + [256] 
    T_max = Config.MAX_TIME_STEPS 
    
    time_slices = [] # Final input array (3 channels * T_max maps)

    for c in range(3): # Channels 0 (R), 1 (G), 2 (B)
        channel = img_rgb[:, :, c]
        
        # Iterate over intervals. Index 'i' corresponds to time (i+1)
        # The interval [0, T1) (darkest) will be the first (i=0, time t=1)
        for i in range(T_max):
            T_low = interval_limits[i]
            T_high = interval_limits[i+1]
            
            # Binary map: 1 if the pixel is in the intensity interval [T_low, T_high)
            ch_map = np.logical_and(channel >= T_low, channel < T_high).astype(int)

            time_slices.append(ch_map.flatten())

    return time_slices

# --- MODIFICATION TO process_dataset ---

def process_dataset(dataset, count): 
    """ Prepares the dataset using temporal encoding. """
    imgs = dataset.imgs[:count]
    labels = dataset.labels[:count].flatten()
    
    red_input = []
    green_input = []
    blue_input = []
    
    thresholds = Config.TEMPORAL_THRESHOLD_LEVELS 
    T_max = Config.MAX_TIME_STEPS 

    for img in imgs:
        # Gets (3 * T_max) binarized arrays (e.g. R_t1, R_t2, ..., B_t5)
        time_slices = temporal_encode_image(img, thresholds) 
        
        # Concatenate the T_max TEMPORAL slices for each channel 
        # R = Slice 0, ..., T_max-1
        red_input.append(np.concatenate(time_slices[0 : T_max]))
        # G = Slice T_max, ..., 2*T_max-1
        green_input.append(np.concatenate(time_slices[T_max : 2 * T_max]))
        # B = Slice 2*T_max, ..., 3*T_max-1
        blue_input.append(np.concatenate(time_slices[2 * T_max : 3 * T_max]))

    return (
        np.array(red_input),
        np.array(green_input),
        np.array(blue_input),
        labels
    )

# ----------------------------------------------------------------------
# --- SNPS LOGIC: Dedicated Pooling and Reshape Correction ---
# ----------------------------------------------------------------------

def SNPS_csv(threshold_matrix=None, filename=None):
    """
    Generates the SNPS. Adapted to connect the N L1 neurons (the N darkness levels)
    to their dedicated L2 neuron set (Dedicated Pooling).
    """
    if filename is None:
        filename = os.path.join(project_root, "csv", Config.CSV_NAME)
        
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules", "forget_rule"])

        # Layer 1: Input (NEURONS_LAYER1)
        img_pixels_old = Config.IMG_SHAPE**2 # 784
        num_levels = Config.NUM_INPUT_LEVELS # N (e.g. 5)
        
        # Calculate the number of L2 neurons in a single spatial level (e.g. 196)
        neurons_per_level = Config.NEURONS_LAYER2 // num_levels 

        for neuron_id in range(Config.NEURONS_LAYER1): 
            # 1. Calculate the original pixel index (2D position 0-783)
            pixel_index = neuron_id % img_pixels_old 
            
            # 2. Calculate the level index (0 to N-1)
            level_index = neuron_id // img_pixels_old 
            
            # 3. Calculate the block ID (2D block position 0-195)
            r = pixel_index // Config.IMG_SHAPE
            c = pixel_index % Config.IMG_SHAPE
            block_row = r // Config.BLOCK_SHAPE
            block_col = c // Config.BLOCK_SHAPE
            block_id = block_row * (Config.IMG_SHAPE // Config.BLOCK_SHAPE) + block_col
            
            # 4. Calculate the dedicated Layer 2 (L2) neuron ID 
            # L2_ID = Start_L2 + (Level_Offset * Neurons_per_Level) + Physical_Position
            base_l2_start = Config.NEURONS_LAYER1
            level_offset = level_index * neurons_per_level 
            output_neuron = base_l2_start + level_offset + block_id
            
            # L1 neurons connect to their dedicated L2 neuron
            writer.writerow([
                neuron_id, 0, f"[{output_neuron}]", 0, "[0,1,1,1,0]" 
            ])

        # Layer 2: Pooling (NEURONS_LAYER2 neurons) - Starts from ID NEURONS_LAYER1
        output_targets = str(list(range(Config.NEURONS_LAYER1_2, Config.NEURONS_TOTAL))) 
        
        if threshold_matrix is None:
            # Phase 1: Initial rule [1, 1, 0, 1, 0]
            for neuron_id in range(Config.NEURONS_LAYER1, Config.NEURONS_LAYER1_2):
                  writer.writerow([
                      neuron_id, 0, output_targets, 1, "[1, 1, 0, 1, 0]",
                      "[1,1,1,0,0]"
                  ])
        else: 
            # Phase 2: Rule learned from normalize_rules
            threshold_array = threshold_matrix.flatten()
            for neuron_id in range(Config.NEURONS_LAYER1, Config.NEURONS_LAYER1_2):
                firing_threshold = threshold_array[neuron_id - Config.NEURONS_LAYER1]
                # The rule remains [1, T, 0, 1, 0] (Fire if T, Full reset)
                firing_rule = f"[1,{firing_threshold},0,1,0]"
                writer.writerow([
                    neuron_id, 0, output_targets, 1, firing_rule, "[1,1,1,0,0]"
                ])

        # Layer 3: Output (8 neurons)
        for neuron_id in range(Config.NEURONS_LAYER1_2, Config.NEURONS_TOTAL):
            writer.writerow([
                neuron_id, 0, "[]", 2, "[1,1,1,0,0]"
            ])

def normalize_rules(firing_counts, imgs_number):
    """
    Used in the rule training phase.
    Updated for Dedicated Pooling.
    """
    min_threshold = 1
    
    # max_threshold is just BLOCK_SHAPE**2 for Dedicated Pooling
    max_threshold = Config.BLOCK_SHAPE**2 
    
    # Calculate normalized firing frequency
    norm = firing_counts / imgs_number
    
    # Scale normalized frequency between min and max threshold
    threshold_matrix = norm * (max_threshold - min_threshold) + min_threshold
    
    # Round and convert to integers
    threshold_matrix = np.round(threshold_matrix).astype(int)
    
    # Ensure threshold is at least 1
    threshold_matrix[threshold_matrix < 1] = 1 
    
    SNPS_csv(threshold_matrix)


# ----------------------------------------------------------------------
# --- TRAIN AND TEST LOGIC (Reshape Correction) ---
# ----------------------------------------------------------------------

def rules_train_SNPS(spike_train):
    # Simulation lasts MAX_TIME_STEPS to collect temporal evidence
    snps = SNPSystem(Config.MAX_TIME_STEPS, Config.TRAIN_SIZE + Config.MAX_TIME_STEPS, "images", "prediction", True)
    
    csv_path = os.path.join(project_root, "csv", Config.CSV_NAME)
    snps.load_neurons_from_csv(csv_path)
    
    snps.spike_train = spike_train
    snps.layer_2_firing_counts = np.zeros(Config.NEURONS_LAYER2, dtype=int)
    w, e = snps.start()
    update_energy(w, e)

    # --- CRITICAL CORRECTION: 3D RESHAPE ---
    rows_cols = int(Config.IMG_SHAPE / Config.BLOCK_SHAPE) # 14
    levels = Config.NUM_INPUT_LEVELS # 5
    
    # Reshape to (Levels, Rows, Cols) (e.g. 5, 14, 14)
    reshaped_counts = snps.layer_2_firing_counts.reshape((levels, rows_cols, rows_cols))
    
    normalize_rules(reshaped_counts, Config.TRAIN_SIZE)

def syn_train_SNPS(spike_train, labels):
    # Simulation lasts MAX_TIME_STEPS to collect temporal evidence
    snps = SNPSystem(Config.MAX_TIME_STEPS, Config.TRAIN_SIZE + Config.MAX_TIME_STEPS, "images", "prediction", True)
    
    csv_path = os.path.join(project_root, "csv", Config.CSV_NAME)
    snps.load_neurons_from_csv(csv_path)
    
    snps.spike_train = spike_train
    snps.layer_2_synapses = np.zeros((Config.CLASSES, Config.NEURONS_LAYER2), dtype=float) # matrix for destroy synapses
    snps.labels = labels
    snps.layer_2_firing_counts = np.zeros(Config.NEURONS_LAYER2, dtype=int)
    w, e = snps.start()
    update_energy(w, e)

    pruned_matrix = prune_matrix(snps.layer_2_synapses)
    prune_SNPS(pruned_matrix)

def compute_SNPS(spike_train):
    # Simulation lasts MAX_TIME_STEPS
    snps = SNPSystem(Config.MAX_TIME_STEPS, Config.TEST_SIZE + Config.MAX_TIME_STEPS, "images", "prediction", True)
    
    csv_path = os.path.join(project_root, "csv", Config.CSV_NAME_PRUNED)
    snps.load_neurons_from_csv(csv_path)
    
    snps.spike_train = spike_train
    snps.layer_2_firing_counts = np.zeros(Config.NEURONS_LAYER2, dtype=int)
    w, e = snps.start()
    update_energy(w, e)

    # Returns total spike accumulation in T_max steps (Frequency Encoding)
    return snps.output_array[3:-2] 

def configure_custom_snps():
    """Configures specific parameters for this experiment"""
    # Set correct CSV names
    Config.CSV_NAME = "CSV_NAME_T"
    Config.CSV_NAME_PRUNED = "CSV_NAME_T_PRUNED"

    # If parameters were not set by main.py (configure("temporal")), use defaults
    # if Config.TEMPORAL_THRESHOLD_LEVELS is None:
    #     print("⚠️ ATTENZIONE: Configurazione 'temporal' non rilevata. Uso parametri di default.")
    #     Config.BLOCK_SHAPE = 3
    #     Config.TEMPORAL_THRESHOLD_LEVELS = [50, 100, 150, 200]
    #     Config.MAX_TIME_STEPS = len(Config.TEMPORAL_THRESHOLD_LEVELS) + 1
    #     Config.NUM_INPUT_LEVELS = Config.MAX_TIME_STEPS
        
    #     Config.NEURONS_LAYER1 = int(Config.IMG_SHAPE ** 2) * Config.NUM_INPUT_LEVELS
    #     # L2 Neurons = (Blocks) * (Temporal Levels)
    #     Config.NEURONS_LAYER2 = int((Config.IMG_SHAPE / Config.BLOCK_SHAPE) ** 2) * Config.NUM_INPUT_LEVELS
    #     Config.NEURONS_LAYER1_2 = int(Config.NEURONS_LAYER1 + Config.NEURONS_LAYER2)
    #     Config.NEURONS_TOTAL = Config.NEURONS_LAYER1_2 + Config.CLASSES


def launch_SNPS():
    """main method of the class, manage all the SN P systems"""
    configure_custom_snps()
    
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



def prune_matrix(synapses):
    """prepare the matrix for the synapses"""
    keep_matrix = np.zeros_like(synapses, dtype=int) 
    for class_idx in range(synapses.shape[0]):
        weights = synapses[class_idx]
        N = len(weights)
        
        # Calculate quantities
        num_pruned = int(Config.PRUNING_PERC * N)
        num_inhibit = int(Config.INHIBIT_PERC * N)
        num_excite = int((1 - Config.PRUNING_PERC - Config.INHIBIT_PERC) * N)
        
        sorted_indices = np.argsort(weights)
        
        # 1. Inhibitors: The lowest weights (Bottom)
        inhibit_indices = sorted_indices[:num_inhibit]
        keep_matrix[class_idx, inhibit_indices] = -1
        
        # 2. Pruned: The middle weights (Remain 0)
        
        # 3. Excitators: The highest weights (Top)
        excite_indices = sorted_indices[-num_excite:]
        keep_matrix[class_idx, excite_indices] = 1
        
    return keep_matrix

def prune_SNPS(pruned_matrix):
    """change the synapses in the csv file"""
    csv_in = os.path.join(project_root, "csv", Config.CSV_NAME)
    csv_out = os.path.join(project_root, "csv", Config.CSV_NAME_PRUNED)
    
    with open(csv_in, 'r') as f_in, open(csv_out, 'w', newline='') as f_out:
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
    # 2) Combined ranking logic
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