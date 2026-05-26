"""
deep CNN -> SN P system CSV importer.

Expected architecture:
    Input 28x28x1
    -> Conv1 3x3 valid, 8 filters
    -> Pool1 2x2
    -> Conv2 3x3 valid, 16 filters, connected to all Pool1 feature maps
    -> Pool2 2x2
    -> Output 10

Expected JSON keys:
    conv1_kernels : shape (8, 1, 3, 3)
    conv2_kernels : shape (16, 8, 3, 3)
    fc_weights    : shape (10, 400)

For backward compatibility, conv1_kernels may also be named conv_kernels.
All weights are expected to be ternary {-1, 0, +1}.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import numpy as np


try:
    from handle_csv import _build_layer1_qrange_rules, _with_negative_forgetting
except ImportError:
    def _with_negative_forgetting(rules: list[str]) -> list[str]:
        """Fallback: forget any negative charge exactly."""
        return rules + [f"[0,{-i},{-i},0,0]" for i in range(1, 200)]

    def _build_layer1_qrange_rules() -> list[str]:
        """Fallback input rules for charges 1..Q_RANGE."""
        return [f"[0,{i},{i},{i},0]" for i in range(255, 0, -1)]


@dataclass(frozen=True)
class DeepConfig:
    img_shape: int = 28
    in_channels: int = 1
    conv1_filters: int = 8
    conv2_filters: int = 16
    kernel_shape: int = 3
    pooling_size: int = 2
    classes: int = 10
    q_range: int = 8
    csv_name: str = "SNPS_deep_cnn.csv"

    @property
    def conv1_shape(self) -> int:
        return self.img_shape - self.kernel_shape + 1

    @property
    def pool1_shape(self) -> int:
        return self.conv1_shape // self.pooling_size

    @property
    def conv2_shape(self) -> int:
        return self.pool1_shape - self.kernel_shape + 1

    @property
    def pool2_shape(self) -> int:
        return self.conv2_shape // self.pooling_size

    @property
    def neurons_l1(self) -> int:
        return self.img_shape * self.img_shape

    @property
    def neurons_conv1_per_map(self) -> int:
        return self.conv1_shape * self.conv1_shape

    @property
    def neurons_pool1_per_map(self) -> int:
        return self.pool1_shape * self.pool1_shape

    @property
    def neurons_conv2_per_map(self) -> int:
        return self.conv2_shape * self.conv2_shape

    @property
    def neurons_pool2_per_map(self) -> int:
        return self.pool2_shape * self.pool2_shape

    @property
    def neurons_conv1(self) -> int:
        return self.conv1_filters * self.neurons_conv1_per_map

    @property
    def neurons_pool1(self) -> int:
        return self.conv1_filters * self.neurons_pool1_per_map

    @property
    def neurons_conv2(self) -> int:
        return self.conv2_filters * self.neurons_conv2_per_map

    @property
    def neurons_pool2(self) -> int:
        return self.conv2_filters * self.neurons_pool2_per_map

    @property
    def offset_conv1(self) -> int:
        return self.neurons_l1

    @property
    def offset_pool1(self) -> int:
        return self.offset_conv1 + self.neurons_conv1

    @property
    def offset_conv2(self) -> int:
        return self.offset_pool1 + self.neurons_pool1

    @property
    def offset_pool2(self) -> int:
        return self.offset_conv2 + self.neurons_conv2

    @property
    def offset_output(self) -> int:
        return self.offset_pool2 + self.neurons_pool2


def _load_deep_weights(json_path: str, cfg: DeepConfig):
    with open(json_path) as f:
        data = json.load(f)

    conv1_key = "conv1_kernels" if "conv1_kernels" in data else "conv_kernels"
    conv1 = np.asarray(data[conv1_key], dtype=int)
    conv2 = np.asarray(data["conv2_kernels"], dtype=int)
    fc_q = np.asarray(data["fc_weights"], dtype=int).T

    expected_conv1 = (cfg.conv1_filters, cfg.in_channels, cfg.kernel_shape, cfg.kernel_shape)
    expected_conv2 = (cfg.conv2_filters, cfg.conv1_filters, cfg.kernel_shape, cfg.kernel_shape)
    expected_fc = (cfg.neurons_pool2, cfg.classes)

    if conv1.shape != expected_conv1:
        raise ValueError(f"{conv1_key} shape must be {expected_conv1}, got {conv1.shape}")
    if conv2.shape != expected_conv2:
        raise ValueError(f"conv2_kernels shape must be {expected_conv2}, got {conv2.shape}")
    if fc_q.shape != expected_fc:
        raise ValueError(f"fc_weights.T shape must be {expected_fc}, got {fc_q.shape}")

    for name, arr in {"conv1": conv1, "conv2": conv2, "fc": fc_q}.items():
        bad = sorted(set(arr.flatten()) - {-1, 0, 1})
        if bad:
            raise ValueError(f"{name} contains non-ternary values: {bad}")

    return conv1, conv2, fc_q


def _build_exact_forward_rules(max_charge: int) -> list[str]:
    rules = [f"[0,{i},{i},{i},0]" for i in range(max_charge, 0, -1)]
    return _with_negative_forgetting(rules)


def _build_pooling_rules(max_incoming_charge: int) -> list[str]:
    rules = []
    for charge in range(max_incoming_charge, 0, -1):
        out_spikes = charge // 4
        if out_spikes > 0:
            rules.append(f"[1,{charge},{charge},{out_spikes},0]")
    return _with_negative_forgetting(rules)


def _compute_max_charges(conv1: np.ndarray, conv2: np.ndarray, cfg: DeepConfig):
    max_conv1 = []
    max_pool1_in = []
    max_pool1_out = []

    for k in range(cfg.conv1_filters):
        positive = int(np.sum(conv1[k] == 1))
        conv_max = positive * cfg.q_range
        max_conv1.append(conv_max)
        max_pool1_in.append(conv_max * 4)
        max_pool1_out.append(conv_max)

    max_conv2 = []
    max_pool2_in = []

    for out_k in range(cfg.conv2_filters):
        conv_max = 0
        for in_k in range(cfg.conv1_filters):
            positive = int(np.sum(conv2[out_k, in_k] == 1))
            conv_max += positive * max_pool1_out[in_k]
        max_conv2.append(conv_max)
        max_pool2_in.append(conv_max * 4)

    return {
        "conv1": max_conv1,
        "pool1_in": max_pool1_in,
        "conv2": max_conv2,
        "pool2_in": max_pool2_in,
    }


def _pool_target(local_idx: int, in_shape: int, out_shape: int, pool: int) -> int:
    row = local_idx // in_shape
    col = local_idx % in_shape
    return (row // pool) * out_shape + (col // pool)


def SNPS_csv_from_deep_cnn(
        json_path: str = "snps_weights_deep.json",
        output_dir: str = "csv",
        cfg: DeepConfig = DeepConfig(),
) -> str:
    conv1, conv2, fc_q = _load_deep_weights(json_path, cfg)
    max_charges = _compute_max_charges(conv1, conv2, cfg)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, cfg.csv_name)

    with open(out_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        l1_rules = _build_layer1_qrange_rules()

        # Layer 1: input pixels -> Conv1.
        for neuron_id in range(cfg.neurons_l1):
            i_row = neuron_id // cfg.img_shape
            i_col = neuron_id % cfg.img_shape
            output_targets = []

            for k1 in range(cfg.conv1_filters):
                map_offset = cfg.offset_conv1 + k1 * cfg.neurons_conv1_per_map
                kernel = conv1[k1, 0]

                for ki in range(cfg.kernel_shape):
                    for kj in range(cfg.kernel_shape):
                        o_row = i_row - ki
                        o_col = i_col - kj
                        if 0 <= o_row < cfg.conv1_shape and 0 <= o_col < cfg.conv1_shape:
                            weight = int(kernel[ki, kj])
                            if weight == 0:
                                continue
                            target = map_offset + o_row * cfg.conv1_shape + o_col
                            output_targets.append(target if weight == 1 else -target)

            writer.writerow([neuron_id, 0, str(output_targets), 0, *l1_rules])

        # Layer 2: Conv1 -> Pool1.
        for k1 in range(cfg.conv1_filters):
            rules = _build_exact_forward_rules(max_charges["conv1"][k1])
            map_offset = cfg.offset_conv1 + k1 * cfg.neurons_conv1_per_map
            pool_offset = cfg.offset_pool1 + k1 * cfg.neurons_pool1_per_map

            for i in range(cfg.neurons_conv1_per_map):
                pool_local = _pool_target(i, cfg.conv1_shape, cfg.pool1_shape, cfg.pooling_size)
                writer.writerow([map_offset + i, 0, str([pool_offset + pool_local]), 1, *rules])

        # Layer 3: Pool1 -> Conv2.
        for in_k in range(cfg.conv1_filters):
            rules = _build_pooling_rules(max_charges["pool1_in"][in_k])
            map_offset = cfg.offset_pool1 + in_k * cfg.neurons_pool1_per_map

            for neuron_local in range(cfg.neurons_pool1_per_map):
                p_row = neuron_local // cfg.pool1_shape
                p_col = neuron_local % cfg.pool1_shape
                output_targets = []

                for out_k in range(cfg.conv2_filters):
                    conv2_map_offset = cfg.offset_conv2 + out_k * cfg.neurons_conv2_per_map
                    kernel = conv2[out_k, in_k]

                    for ki in range(cfg.kernel_shape):
                        for kj in range(cfg.kernel_shape):
                            o_row = p_row - ki
                            o_col = p_col - kj
                            if 0 <= o_row < cfg.conv2_shape and 0 <= o_col < cfg.conv2_shape:
                                weight = int(kernel[ki, kj])
                                if weight == 0:
                                    continue
                                target = conv2_map_offset + o_row * cfg.conv2_shape + o_col
                                output_targets.append(target if weight == 1 else -target)

                writer.writerow([map_offset + neuron_local, 0, str(output_targets), 1, *rules])

        # Layer 4: Conv2 -> Pool2.
        for out_k in range(cfg.conv2_filters):
            rules = _build_exact_forward_rules(max_charges["conv2"][out_k])
            map_offset = cfg.offset_conv2 + out_k * cfg.neurons_conv2_per_map
            pool_offset = cfg.offset_pool2 + out_k * cfg.neurons_pool2_per_map

            for i in range(cfg.neurons_conv2_per_map):
                pool_local = _pool_target(i, cfg.conv2_shape, cfg.pool2_shape, cfg.pooling_size)
                writer.writerow([map_offset + i, 0, str([pool_offset + pool_local]), 1, *rules])

        # Layer 5: Pool2 -> Output.
        for out_k in range(cfg.conv2_filters):
            rules = _build_pooling_rules(max_charges["pool2_in"][out_k])
            map_offset = cfg.offset_pool2 + out_k * cfg.neurons_pool2_per_map

            for local_i in range(cfg.neurons_pool2_per_map):
                flat_i = out_k * cfg.neurons_pool2_per_map + local_i
                output_targets = []

                for class_j in range(cfg.classes):
                    weight = int(fc_q[flat_i, class_j])
                    if weight == 0:
                        continue
                    target = cfg.offset_output + class_j
                    output_targets.append(target if weight == 1 else -target)

                writer.writerow([map_offset + local_i, 0, str(output_targets), 1, *rules])

        # Output neurons.
        output_rules = _with_negative_forgetting(["[1,1,0,0,0]"])
        for class_j in range(cfg.classes):
            writer.writerow([cfg.offset_output + class_j, 0, "[]", 2, *output_rules])

    print(f"CSV written -> {out_path}")
    print(f"Input:  {cfg.neurons_l1}")
    print(f"Conv1:  {cfg.neurons_conv1} ({cfg.conv1_shape}x{cfg.conv1_shape}x{cfg.conv1_filters})")
    print(f"Pool1:  {cfg.neurons_pool1} ({cfg.pool1_shape}x{cfg.pool1_shape}x{cfg.conv1_filters})")
    print(f"Conv2:  {cfg.neurons_conv2} ({cfg.conv2_shape}x{cfg.conv2_shape}x{cfg.conv2_filters})")
    print(f"Pool2:  {cfg.neurons_pool2} ({cfg.pool2_shape}x{cfg.pool2_shape}x{cfg.conv2_filters})")
    print(f"Output: {cfg.classes}")
    return out_path


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, "cnn_twin_results", "deep_snps_weights.json")
    output_dir = os.path.join(project_root, "csv")

    SNPS_csv_from_deep_cnn(json_path=json_path, output_dir=output_dir)
