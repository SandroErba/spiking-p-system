import csv
from sps import Config


def quantized_SNPS_csv(filename="csv/" + Config.CSV_NAME_Q):
    """Generate the SN P system to analyze chosen images
    If a matrix is passed, update the existing P system"""
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # Layer 1: Input RGB from 28x28 to 14x14 using 2x2 blocks
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
                "[0,4,4,4,0]",
                "[0,3,3,3,0]",
                "[0,2,2,2,0]",
                "[0,1,1,1,0]"         # firing rule #boolean is "[0,1,1,1,0]"
            ])

        # Layer 2: Pooling (49 neurons)
        output_targets = str(list(range(Config.NEURONS_LAYER1_2, Config.NEURONS_TOTAL))) # for firing at the output neurons
        for neuron_id in range(Config.NEURONS_LAYER1, Config.NEURONS_LAYER1_2):
            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                output_targets,       # output_targets
                1,                    # neuron_type
                "[1,13,1,4,0]",
                "[1,9,1,3,0]",
                "[1,5,1,2,0]",
                "[1,1,1,1,0]",       # firing rules if c >= 1
                "[1,1,1,0,0]"        # forgetting rule if didn't fire
            ])

        # Layer 3: Output (8 neurons)
        for neuron_id in range(Config.NEURONS_LAYER1_2, Config.NEURONS_TOTAL):
            #label = neuron_id - Config.NEURONS_LAYER1_2
            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                "[]",                 # output_targets
                2,                    # neuron_type
                "[1,1,1,0,0]"         # forgetting rule
            ])


def binarized_SNPS_csv(threshold_matrix=None, filename="csv/" + Config.CSV_NAME_B):
    """Generate the SN P system to analyze chosen images
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

def prune_SNPS(pruned_matrix):
    """change the synapses in the csv file"""
    csv_name = Config.CSV_NAME_Q if Config.QUANTIZATION else Config.CSV_NAME_B
    csv_name_pruned = Config.CSV_NAME_Q_PRUNED if Config.QUANTIZATION else Config.CSV_NAME_B_PRUNED
    with open("csv/" + csv_name, 'r') as f_in, open("csv/" + csv_name_pruned, 'w', newline='') as f_out:
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


def kernel_SNPS_csv():
    """
    Generate a 3-layer SN P system to perform edge detection on a 28x28 image
    using 6 convolution kernels (2x2) with values 1 and -1.
    The structure is:
    - Layer 1: Input neurons (784 neurons), firing to 6 parallel subnetworks
    - Layer 2: One 27x27 grid per kernel (6 kernels → 4374 neurons)
    - Layer 3: 27x27 neurons (729 neurons), sum of all filtered maps
    for more info see: Ultrafast neuromorphic photonic image processing with a VCSEL neuron"""
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

    with open("csv/" + Config.CSV_KERNEL_NAME, mode='w', newline='') as csv_file:
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