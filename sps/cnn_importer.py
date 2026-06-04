"""
CNN Importer
=================
Generates the SN P system CSV from trained CNN weights stored in snps_weights.json.

Replaces cnn_SNPS_csv() + extend_csv() with two new functions:
  - SNPS_csv_from_cnn()   : builds layers 1-3 using trained conv kernels
  - extend_csv_from_cnn()     : extends CSV with trained FC weights (L3 → output)

Key differences from the original:
  - Conv kernels come from snps_weights.json["conv_kernels"] instead of Config.KERNELS
  - K_RANGE is computed automatically per kernel from the number of +1 weights
  - max_L2[k] = count(+1 in kernel[k]) * Q_RANGE
  - max_L3[k] = max_L2[k] * 4   (before the internal /4 firing rule)
  - No multipliers/importance scoring (CNN training handles this implicitly)
  - FC weights come from snps_weights.json["fc_weights"], shape (10, 1352) → transposed to (1352, 10)
"""

import csv
import json
import os
import numpy as np

from handle_csv import _build_layer1_qrange_rules, _with_negative_forgetting
from sps.config import Config


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_cnn_weights(json_path: str) -> tuple[list, np.ndarray]:
    """
    Load quantized {-1, 0, +1} weights from snps_weights.json.

    Returns:
        kernels : list of 8 kernels, each a 3x3 list of ints ({-1, 0, +1})
        fc_q    : np.array of shape (1352, 10)  [transposed from (10, 1352)]
    """
    with open(json_path) as f:
        data = json.load(f)

    # conv_kernels shape in JSON: (8, 1, 3, 3)
    raw_kernels = np.array(data["conv_kernels"])  # (8, 1, 3, 3)
    kernels = []
    for k in range(raw_kernels.shape[0]):
        kernel_2d = raw_kernels[k, 0].tolist()    # (3, 3) → list of lists
        kernels.append(kernel_2d)

    # fc_weights shape in JSON: (10, 1352) → transpose to (1352, 10)
    fc_q = np.array(data["fc_weights"]).T         # (1352, 10)

    print("cnn weights loaded")
    return kernels, fc_q


def _compute_max_charges(kernels: list, q_range: int) -> dict:
    """
    Compute per-kernel maximum charges across all three layers.

    Args:
        kernels : list of 8 kernels, each 3x3 with values in {-1, 0, +1}
        q_range : quantization range (e.g. 8)

    Returns dict with per-kernel values:
        max_L2[k] = count(+1 in kernel[k]) * q_range
        max_L3[k] = max_L2[k] * 4
    """
    max_L2 = {}
    max_L3 = {}

    for k_index, kernel in enumerate(kernels):
        positive_synapses = sum(
            1 for row in kernel for val in row if val == 1
        )

        if positive_synapses == 0:
            # Edge case: kernel has no +1 weights → neurons never fire positively.
            # Set minimum of 1 so rule loops don't break.
            print(f"  Warning: kernel {k_index} has no +1 weights. "
                  f"Layer 2 neurons in this block will never fire positively.")
            positive_synapses = 1

        max_L2[k_index] = positive_synapses * q_range
        max_L3[k_index] = max_L2[k_index] * 4

        print(f"  Kernel {k_index}: +1 synapses={positive_synapses}, "
              f"max_L2={max_L2[k_index]}, max_L3={max_L3[k_index]}")

    print("max charges computed")
    return {"max_L2": max_L2, "max_L3": max_L3}


def _build_layer2_rules_cnn(max_L2_k: int) -> list:
    """
    Layer-2 forwarding rules for kernel k, using computed max charge.
    Identical logic to _build_layer2_rules() but uses max_L2_k instead of Config.K_RANGE.

    Each rule fires exactly `i` spikes when charge is exactly `i`.
    Covers all values from max_L2_k down to 1.
    """
    rules = []
    for i in range(max_L2_k, 0, -1):
        rules.append(f"[0,{i},{i},{i},0]")
    return _with_negative_forgetting(rules)


def _build_layer3_rules_cnn(max_L3_k: int) -> list:
    """
    Layer-3 (pooling) rules for kernel k, using computed max charge.

    Each Layer 3 neuron receives input from exactly 4 Layer 2 neurons.
    It fires floor(incoming_charge / 4) spikes.

    Rules cover all incoming values from max_L3_k down to 1.
    For incoming charge `c`, the neuron fires floor(c / 4) spikes.
    Multiple values of c can map to the same output (e.g. 4,5,6,7 → all fire 1).
    We write one rule per distinct output value, triggered at the highest c
    that maps to it, using >= matching (lower bound = c, upper bound = max_L3_k
    for that group — but since SNP rules match exact values, we write one rule
    per incoming value for correctness).
    """
    rules = []
    for c in range(max_L3_k, 0, -1):
        out_spikes = c // 4
        if out_spikes > 0:
            rules.append(f"[1,{c},{c},{out_spikes},0]")
        # c values where floor(c/4) == 0 produce no spikes; handled by forgetting rule
    return _with_negative_forgetting(rules)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION 1: Build layers 1-3 CSV from trained CNN kernels
# ─────────────────────────────────────────────────────────────────────────────

def SNPS_csv_from_cnn(json_path: str = "snps_weights.json"):
    """
    Generate the SN P system CSV using trained CNN conv kernels.
    Covers Layer 1 (input), Layer 2 (conv), Layer 3 (pooling).
    Layer 3 output targets are left empty ([]) — filled by extend_csv_from_cnn().

    Args:
        json_path : path to snps_weights.json produced by snps_twin_cnn.py
    """

    print("load json from", json_path)

    kernels, _ = _load_cnn_weights(json_path)

    print(f"Loaded {len(kernels)} kernels from {json_path}")
    print("Computing per-kernel max charges...")
    max_charges = _compute_max_charges(kernels, Config.Q_RANGE)
    max_L2 = max_charges["max_L2"]
    max_L3 = max_charges["max_L3"]

    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(CSV_OUTPUT_DIR, Config.CSV_CNN_NAME, )

    with open(out_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # ── Layer 1: Input neurons (28×28 = 784) ──────────────────────────────
        # Identical to original except kernels come from CNN weights.
        l1_firing_rules = _build_layer1_qrange_rules()

        for neuron_id in range(Config.NEURONS_L1):
            i_row = neuron_id // Config.IMG_SHAPE
            i_col = neuron_id % Config.IMG_SHAPE
            output_targets = []

            for k_index, kernel in enumerate(kernels):
                layer2_offset = Config.NEURONS_L1 + k_index * Config.NEURONS_FEATURE

                for ki in range(Config.KERNEL_SHAPE):
                    for kj in range(Config.KERNEL_SHAPE):
                        o_row = i_row - ki
                        o_col = i_col - kj

                        if 0 <= o_row < Config.SHAPE_FEATURE and 0 <= o_col < Config.SHAPE_FEATURE:
                            output_idx = o_row * Config.SHAPE_FEATURE + o_col
                            target_id = layer2_offset + output_idx
                            weight = kernel[ki][kj]
                            if weight == 1:
                                output_targets.append(target_id)
                            elif weight == -1:
                                output_targets.append(-target_id)
                            # weight == 0: no synapse, skip

            writer.writerow([
                neuron_id,
                0,
                str(output_targets),
                0,
                *l1_firing_rules
            ])

        # ── Layer 2: Conv feature maps (26×26×8 = 5408) ───────────────────────
        for k_index, kernel in enumerate(kernels):
            l2_firing_rules = _build_layer2_rules_cnn(max_L2[k_index])
            layer2_offset = Config.NEURONS_L1 + k_index * Config.NEURONS_FEATURE

            for i in range(Config.NEURONS_FEATURE):
                # Each L2 neuron connects to exactly one L3 pooling neuron
                j = (
                            (i // Config.SHAPE_FEATURE) // Config.POOLING_SIZE
                    ) * Config.SHAPE_POOL + (
                            (i % Config.SHAPE_FEATURE) // Config.POOLING_SIZE
                    )
                target_l3 = (
                        Config.NEURONS_L1
                        + Config.NEURONS_L2
                        + k_index * Config.NEURONS_POOL
                        + j
                )
                writer.writerow([
                    layer2_offset + i,
                    0,
                    str([target_l3]),
                    1,
                    *l2_firing_rules
                ])

        # ── Layer 3: Pooling neurons (13×13×8 = 1352) ─────────────────────────
        # Output targets left as [] — will be filled by extend_csv_from_cnn()
        for k_index in range(Config.KERNEL_NUMBER):
            l3_firing_rules = _build_layer3_rules_cnn(max_L3[k_index])
            layer3_offset = Config.NEURONS_L1 + Config.NEURONS_L2 + k_index * Config.NEURONS_POOL

            for i in range(Config.NEURONS_POOL):
                writer.writerow([
                    layer3_offset + i,
                    0,
                    "[]",           # filled later by extend_csv_from_cnn()
                    1,
                    *l3_firing_rules
                ])

    print(f"\nCSV written → {out_path}")
    print(f"  Layer 1: {Config.NEURONS_L1} neurons")
    print(f"  Layer 2: {Config.NEURONS_L2} neurons ({len(kernels)} feature maps)")
    print(f"  Layer 3: {Config.NEURONS_L3} neurons (pooling, targets TBD)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION 2: Extend CSV with trained FC weights (L3 → output)
# ─────────────────────────────────────────────────────────────────────────────

def extend_csv_from_cnn(file_path: str, json_path: str = "snps_weights.json"):
    """
    Extend the CSV produced by SNPS_csv_from_cnn() with:
      - Output targets for each Layer 3 neuron (from FC weights)
      - Updated firing rules for Layer 3 neurons (per-kernel max)
      - New output neurons (10 classes)

    Args:
        file_path : path to CSV produced by SNPS_csv_from_cnn()
        json_path : path to snps_weights.json

    Returns:
        new_file_path : path to the extended CSV
    """
    kernels, fc_q = _load_cnn_weights(json_path)   # fc_q shape: (1352, 10)

    print(f"Loaded FC weights: shape {fc_q.shape}")
    max_charges = _compute_max_charges(kernels, Config.Q_RANGE)
    max_L3 = max_charges["max_L3"]

    # Build output path
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_external{ext}"

    with open(file_path, newline='') as f:
        rows = list(csv.reader(f))

    output_offset = Config.NEURONS_L1 + Config.NEURONS_L2 + Config.NEURONS_L3
    pool_offset   = Config.NEURONS_L1 + Config.NEURONS_L2
    n_classes     = min(Config.CLASSES, fc_q.shape[1])

    # ── Update Layer 3 neurons: output targets + firing rules ─────────────────
    for i in range(Config.NEURONS_L3):
        row = rows[pool_offset + 1 + i]   # +1 for header

        # Which kernel block does neuron i belong to?
        k_index = i // Config.NEURONS_POOL

        # Output targets from FC weights (column i of fc_q = connections to all 10 classes)
        new_targets = []
        for j in range(n_classes):
            weight = fc_q[i, j]
            target_id = output_offset + j
            if weight == 1:
                new_targets.append(target_id)
            elif weight == -1:
                new_targets.append(-target_id)
            # weight == 0: no synapse
        row[2] = str(new_targets)

        # Rebuild firing rules using per-kernel max_L3
        # (replaces the placeholder rules written by SNPS_csv_from_cnn)
        new_rules = _build_layer3_rules_cnn(max_L3[k_index])
        row[:] = row[:4] + new_rules

    # ── Add output neurons (10 classes) ───────────────────────────────────────
    output_rules = _with_negative_forgetting(["[1,1,0,0,0]"])

    for j in range(Config.CLASSES):
        new_row = [
            output_offset + j,
            0,
            "[]",
            2,
            *output_rules
        ]
        rows.append(new_row)

    # ── Write extended CSV ─────────────────────────────────────────────────────
    with open(new_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nExtended CSV written → {new_file_path}")
    print(f"  Layer 3 targets updated: {Config.NEURONS_L3} neurons")
    print(f"  Output neurons added:    {Config.CLASSES} classes")
    return new_file_path


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder where this .py file lives
    CSV_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "csv")
    JSON_PATH = os.path.join(PROJECT_ROOT, "cnn_twin_results", "snps_weights.json")

    print(JSON_PATH)
    # Step 1: generate base CSV from trained conv kernels
    csv_path = SNPS_csv_from_cnn(JSON_PATH)

    # Step 2: extend with trained FC weights and output neurons
    final_csv = extend_csv_from_cnn(file_path=csv_path, json_path=JSON_PATH)

    print(f"\nDone. Final SNPS CSV ready: {final_csv}")



#----------EXTERNAL CNN RESULTS-------------
''' 
svm direct accuracy 0.771
logreg direct accuracy 0.845
SNPS svm accuracy: 0.923
SNPS svm imp accuracy: 0.927
real weights svm accuracy: 0.949
SNPS logreg accuracy: 0.917
SNPS logreg imp accuracy: 0.931
real weights logreg accuracy: 0.948
SNPS ensemble accuracy: 0.936
SNPS imp ensemble accuracy: 0.94

first: 0.344, then i balanced 0.8 zero, 0.1 pos and neg
0.445 con pesi bilanciati per rispecchiare le migliori performance precedenti
0.669 con QAT e senza metodo di quantizzazione, i pesi sono già quantizzati nella rete
0.426 usando soglia a 1.05 (come usata qui) sia per kernels che per ultimo layer
0.798/0.81 usando ternizzazione DOPO il train, prendendola dal codice attuale
0.91/0.912/0.902 con 2 stage approach: prima train completo, ternarizzo e freezo kernel, secondo train su FC

ora: 
    -utilizzare il freeze train su rete più profonda (vedi deep classes)
    -nuovo dataset
'''