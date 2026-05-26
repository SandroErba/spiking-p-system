"""
Deep Twin CNN for Spiking Neural P System (SNPS)
================================================
Architecture mirrors the deeper SNPS:
  Input  : 28x28 grayscale, quantized to [0, Q_RANGE]
  Conv1  : Conv2d(1, 8, kernel_size=3, stride=1, padding=0, bias=False)
  ReLU   : mirrors forgetting of negative charges
  Pool1  : AvgPool2d(2, 2)
  Conv2  : Conv2d(8, 16, kernel_size=3, stride=1, padding=0, bias=False)
  ReLU   : mirrors forgetting of negative charges
  Pool2  : AvgPool2d(2, 2)
  FC     : Linear(400, 10, bias=False)

Shape flow:
  28x28x1 -> 26x26x8 -> 13x13x8 -> 11x11x16 -> 5x5x16 -> 400 -> 10

Ternarized output JSON:
  deep_snps_weights.json

Expected by the SNPS CSV generator:
  conv1_kernels : (8, 1, 3, 3)
  conv2_kernels : (16, 8, 3, 3)
  fc_weights    : (10, 400)
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


Q_RANGE = 10
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
K_THRESHOLD = 1.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms(q_range: int = Q_RANGE):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * q_range),
    ])


class DeepSNPSTwinCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(400, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)          # (B, 8, 26, 26)
        x = self.relu1(x)
        x = self.pool1(x)          # (B, 8, 13, 13)
        x = self.conv2(x)          # (B, 16, 11, 11)
        x = self.relu2(x)
        x = self.pool2(x)          # (B, 16, 5, 5)
        x = x.view(x.size(0), -1)  # (B, 400)
        x = self.fc(x)             # (B, 10)
        return x


def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%")

    return history


def ternarize_fc(w: np.ndarray, k: float = K_THRESHOLD) -> np.ndarray:
    w_q = np.zeros_like(w, dtype=np.int8)

    for r in range(w.shape[0]):
        row = w[r]
        threshold = k * np.mean(np.abs(row))
        w_q[r] = np.where(row > threshold, 1, np.where(row < -threshold, -1, 0))

    return w_q.astype(np.int8)


def ternarize_conv_balanced(w: np.ndarray, keep_ratio: float = 1 / 3) -> np.ndarray:
    """
    Balanced per-output-filter ternarization.

    For Conv1, each filter has 9 weights, so this keeps 3 positive and 3 negative.
    For Conv2, each filter has 72 weights, so this keeps 24 positive and 24 negative.
    """
    w_q = np.zeros_like(w, dtype=np.int8)
    n_filters = w.shape[0]
    n_weights = int(np.prod(w.shape[1:]))
    n_keep = max(1, int(round(n_weights * keep_ratio)))

    for f in range(n_filters):
        flat = w[f].reshape(-1)
        pos_idx = np.argsort(flat)[::-1][:n_keep]
        neg_idx = np.argsort(flat)[:n_keep]

        flat_q = np.zeros(n_weights, dtype=np.int8)
        flat_q[pos_idx] = 1
        flat_q[neg_idx] = -1
        w_q[f] = flat_q.reshape(w.shape[1:])

    return w_q


def ternarize_weights(model, k: float = K_THRESHOLD) -> dict:
    conv1_float = model.conv1.weight.detach().cpu().numpy()
    conv2_float = model.conv2.weight.detach().cpu().numpy()
    fc_float = model.fc.weight.detach().cpu().numpy()

    conv1_q = ternarize_conv_balanced(conv1_float)
    conv2_q = ternarize_conv_balanced(conv2_float)
    fc_q = ternarize_fc(fc_float, k=k)

    print("\n-- Ternarization results --")
    for key, w in [
        ("conv1_kernels", conv1_q),
        ("conv2_kernels", conv2_q),
        ("fc_weights", fc_q),
    ]:
        total = w.size
        pos = int((w == 1).sum())
        neg = int((w == -1).sum())
        zero = int((w == 0).sum())
        print(f"  {key}: +1={pos} ({pos/total*100:.1f}%), "
              f"-1={neg} ({neg/total*100:.1f}%), "
              f"0={zero} ({zero/total*100:.1f}%)")

    return {
        "conv1_kernels": conv1_q,
        "conv2_kernels": conv2_q,
        "fc_weights": fc_q,
    }


def save_model(model, path="deep_snps_twin_cnn.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved -> {path}")


def load_model(path="deep_snps_twin_cnn.pth", device=DEVICE):
    model = DeepSNPSTwinCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded <- {path}")
    return model


def save_quantized_weights(weights: dict, path="deep_snps_weights.json"):
    serializable = {k: v.tolist() for k, v in weights.items()}
    with open(path, "w") as f:
        json.dump(serializable, f)
    print(f"Quantized weights saved -> {path}")


def save_float_weights(model, path="deep_snps_weights_float.npz"):
    weights = {
        "conv1_kernels": model.conv1.weight.detach().cpu().numpy(),
        "conv2_kernels": model.conv2.weight.detach().cpu().numpy(),
        "fc_weights": model.fc.weight.detach().cpu().numpy(),
    }
    np.savez(path, **weights)
    print(f"Float weights saved -> {path}")


def main():
    print(f"Device:             {DEVICE}")
    print(f"Quantization range: Q={Q_RANGE}")
    print(f"Epochs:             {EPOCHS}")
    print(f"LR:                 {LR}, Batch: {BATCH_SIZE}")
    print(f"K threshold (FC):   {K_THRESHOLD}\n")

    transform = get_transforms(Q_RANGE)
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = DeepSNPSTwinCNN()
    print(model)
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    print("-- Phase 1: Float training --")
    history = train(model, train_loader, val_loader)

    save_model(model, "deep_snps_twin_cnn.pth")
    save_float_weights(model, "deep_snps_weights_float.npz")

    print("\n-- Phase 2: Ternarization --")
    print("  Conv strategy: balanced per-output-filter")
    print(f"  FC strategy:   adaptive row-wise threshold (k={K_THRESHOLD})")
    ternary_weights = ternarize_weights(model, k=K_THRESHOLD)
    save_quantized_weights(ternary_weights, "deep_snps_weights.json")

    print("\nDone. Files saved:")
    print("  deep_snps_twin_cnn.pth")
    print("  deep_snps_weights_float.npz")
    print("  deep_snps_weights.json")

    return model, ternary_weights, history


if __name__ == "__main__":
    model, weights, history = main()
