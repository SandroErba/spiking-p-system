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


# Try to disable torch.compile which might be causing issues with _dynamo
os.environ["TORCH_COMPILE_DISABLE"] = "1"

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



def _run_epochs(model, train_loader, val_loader, epochs, optimizer, device, label=""):
    criterion = nn.CrossEntropyLoss()
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
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                v_loss += loss.item() * imgs.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total += imgs.size(0)
        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        print(f"{label}Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%")


def _freeze(layer):
    for p in layer.parameters():
        p.requires_grad = False

def _ternarize_inplace(layer_weight_np, model_layer, device):
    """Snap a conv layer to ternary in-place and freeze it."""
    q = ternarize_conv_balanced(layer_weight_np).astype(np.float32)
    with torch.no_grad():
        model_layer.weight.copy_(torch.tensor(q, device=device))
    _freeze(model_layer)

def train_progressive(model, train_loader, val_loader, device=DEVICE):
    """
    Progressive freeze training:
      Stage 1 (20 ep): train everything in float
      Stage 2 ( 5 ep): freeze+ternarize conv1, re-train conv2+FC
      Stage 3 ( 5 ep): freeze+ternarize conv2, re-init and train FC only (re-init FC)
    """
    model = model.to(device)

    # ── Stage 1: full float ──
    print("\n" + "─"*55)
    print("Stage 1: full float training (all layers)")
    print("─"*55)
    opt = optim.Adam(model.parameters(), lr=LR)
    _run_epochs(model, train_loader, val_loader, EPOCHS, opt, device, "[S1] ")

    # ── Stage 2: freeze+ternarize conv1, train conv2+FC ──
    print("\n" + "─"*55)
    print("Stage 2: conv1 frozen+ternary, train conv2+FC")
    print("─"*55)
    _ternarize_inplace(model.conv1.weight.detach().cpu().numpy(), model.conv1, device)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    _run_epochs(model, train_loader, val_loader, 5, opt, device, "[S2] ")

    # ── Stage 3: freeze+ternarize conv2, re-init and train FC only ──
    print("\n" + "─"*55)
    print("Stage 3: conv1+conv2 frozen+ternary, re-train FC")
    print("─"*55)
    _ternarize_inplace(model.conv2.weight.detach().cpu().numpy(), model.conv2, device)
    nn.init.kaiming_uniform_(model.fc.weight, a=0)  # fresh start for FC
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    _run_epochs(model, train_loader, val_loader, 10, opt, device, "[S3] ")



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

    print("-- Phase 1: Progressive training --")
    train_progressive(model, train_loader, val_loader, device=DEVICE)

    save_model(model, "deep_snps_twin_cnn.pth")
    save_float_weights(model, "deep_snps_weights_float.npz")

    print("\n-- Phase 2: Ternarization --")
    print("  Conv strategy: balanced per-output-filter")
    print(f"  FC strategy:   adaptive row-wise threshold (k={K_THRESHOLD})")
    ternary_weights = {
        "conv1_kernels": model.conv1.weight.detach().cpu().numpy().astype(np.int8),
        "conv2_kernels": model.conv2.weight.detach().cpu().numpy().astype(np.int8),
        "fc_weights":    ternarize_fc(model.fc.weight.detach().cpu().numpy(), k=K_THRESHOLD),
    }

    print("\n-- Ternarization results --")
    for key, w in ternary_weights.items():
        total = w.size
        pos = int((w == 1).sum())
        neg = int((w == -1).sum())
        zero = int((w == 0).sum())
        print(f"  {key}: +1={pos} ({pos/total*100:.1f}%), "
              f"-1={neg} ({neg/total*100:.1f}%), "
              f"0={zero} ({zero/total*100:.1f}%)")


    save_quantized_weights(ternary_weights, "deep_snps_weights.json")


    # ── Diagnostics ──
    float_data = np.load("deep_snps_weights_float.npz")
    fc_float = torch.tensor(float_data["fc_weights"].astype(np.float32)).to(DEVICE)

    with torch.no_grad():
        model.conv1.weight.copy_(torch.tensor(ternary_weights["conv1_kernels"].astype(np.float32)).to(DEVICE))
        model.conv2.weight.copy_(torch.tensor(ternary_weights["conv2_kernels"].astype(np.float32)).to(DEVICE))
        model.fc.weight.copy_(fc_float)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += imgs.size(0)
    print(f"Ternary conv + float FC accuracy: {correct/total:.4f}")

    with torch.no_grad():
        model.fc.weight.copy_(torch.tensor(ternary_weights["fc_weights"].astype(np.float32)).to(DEVICE))

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += imgs.size(0)
    print(f"Fully ternary PyTorch accuracy: {correct/total:.4f}")

    return model, ternary_weights,


if __name__ == "__main__":
    model, weights = main()


