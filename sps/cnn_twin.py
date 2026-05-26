"""
Twin CNN for Spiking Neural P System (SNPS)
============================================
Architecture mirrors the SNPS exactly:
  Input  : 28x28 grayscale, quantized to [0, Q_RANGE]
  Layer 1: Conv2d(1, 8, kernel_size=3, stride=1, padding=0, bias=False)
  Layer 2: ReLU  (mirrors the "forgetting" rule: ignore negative charges)
  Layer 3: AvgPool2d(2, 2)  (mirrors floor(sum/4))
  Layer 4: Flatten -> Linear(1352, 10, bias=False)
  Output : raw logits -> argmax for classification

Ternarization strategy (applied AFTER training, not during):
  Conv kernels : balanced per-filter split → ~33% +1, ~33% -1, ~33% 0
                 (only 9 weights per filter, adaptive threshold is unstable at this scale)
  FC weights   : adaptive column-wise threshold → k * mean(|col|)
                 (same method proven to work with SVM/LogReg — ternarize_threshold())
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
import os

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────
Q_RANGE    = 10      # Quantization range: 0-255 → 0-Q_RANGE (paper uses q=8)
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3
K_THRESHOLD = 1.05  # same value used for SVM/LogReg ternarization
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# INPUT QUANTIZATION
# ─────────────────────────────────────────────
def get_transforms(q_range: int = Q_RANGE):
    """Scale input to [0, q_range] to match SNPS quantization."""
    return transforms.Compose([
        transforms.ToTensor(),                      # [0, 1]
        transforms.Lambda(lambda x: x * q_range),  # [0, Q_RANGE]
    ])


# ─────────────────────────────────────────────
# MODEL — plain float CNN, no QAT complications
# ─────────────────────────────────────────────
class SNPSTwinCNN(nn.Module):
    """
    Exact structural twin of the Spiking Neural P System.
    Trained with full float weights — ternarization applied after training.

    Layer correspondence:
      self.conv  ↔  SNPS Layer 1→2 (8 kernels, 3x3, stride 1, no pad)
      ReLU       ↔  SNPS forgetting rule (discard negative charges)
      self.pool  ↔  SNPS Layer 2→3 (average pool 2x2)
      self.fc    ↔  SNPS Layer 3→Output (1352 → 10)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc   = nn.Linear(1352, 10, bias=False)

    def forward(self, x):
        x = self.conv(x)            # (B, 8, 26, 26)
        x = self.relu(x)            # (B, 8, 26, 26)
        x = self.pool(x)            # (B, 8, 13, 13)
        x = x.view(x.size(0), -1)  # (B, 1352)
        x = self.fc(x)             # (B, 10)
        return x


# ─────────────────────────────────────────────
# TRAINING — standard float training
# ─────────────────────────────────────────────
def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss   = criterion(logits, labels)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += imgs.size(0)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%")

    return history


# ─────────────────────────────────────────────
# TERNARIZATION — applied after training
# ─────────────────────────────────────────────
def ternarize_fc(w: np.ndarray, k: float = K_THRESHOLD) -> np.ndarray:
    """
    Adaptive column-wise ternarization for FC weights.
    Identical to the method used for SVM/LogReg — proven to work.

    For each output neuron (row), threshold = k * mean(|row|).
    Values above threshold → +1, below -threshold → -1, else 0.

    Args:
        w : float array of shape (10, 1352)
        k : threshold multiplier (default 1.05, same as SVM/LogReg)

    Returns:
        int8 array of shape (10, 1352) with values in {-1, 0, +1}
    """
    w_q      = np.zeros_like(w, dtype=np.int8)
    n_rows   = w.shape[0]

    for r in range(n_rows):
        row = w[r]
        t   = k * np.mean(np.abs(row))
        w_q[r] = np.where(row > t, 1, np.where(row < -t, -1, 0)).astype(np.int8)

    return w_q


def ternarize_conv(w: np.ndarray) -> np.ndarray:
    """
    Balanced per-filter ternarization for conv kernels.
    Each filter (3x3 = 9 weights) gets ~33% +1, ~33% -1, ~33% 0.

    We cannot use the adaptive method here because with only 9 weights
    per filter the mean(|w|) statistic is too noisy and produces
    heavily skewed results (as seen in experiments).

    Strategy: for each filter, keep top-3 positive and top-3 negative
    weights by magnitude, set rest to 0.

    Args:
        w : float array of shape (8, 1, 3, 3)

    Returns:
        int8 array of shape (8, 1, 3, 3) with values in {-1, 0, +1}
    """
    w_q        = np.zeros_like(w, dtype=np.int8)
    n_filters  = w.shape[0]
    n_weights  = w.shape[1] * w.shape[2] * w.shape[3]  # 1*3*3 = 9
    n_keep     = max(1, n_weights // 3)                 # keep top-3 pos and top-3 neg

    for i in range(n_filters):
        flat     = w[i].flatten()                        # 9 values
        pos_idx  = np.argsort(flat)[::-1][:n_keep]      # top-3 most positive
        neg_idx  = np.argsort(flat)[:n_keep]             # top-3 most negative
        flat_q   = np.zeros(n_weights, dtype=np.int8)
        flat_q[pos_idx] =  1
        flat_q[neg_idx] = -1
        w_q[i]   = flat_q.reshape(w.shape[1:])

    return w_q


def ternarize_weights(model, k: float = K_THRESHOLD) -> dict:
    """
    Extract and ternarize all weights from a trained float model.

    Returns dict with:
        conv_kernels : int8 array (8, 1, 3, 3)  — balanced per-filter split
        fc_weights   : int8 array (10, 1352)    — adaptive column-wise threshold
    """
    conv_float = model.conv.weight.detach().cpu().numpy()  # (8, 1, 3, 3)
    fc_float   = model.fc.weight.detach().cpu().numpy()    # (10, 1352)

    conv_q = ternarize_conv(conv_float)
    fc_q   = ternarize_fc(fc_float, k=k)

    # ── Print stats ──
    print("\n── Ternarization results ──")
    for key, w in [("conv_kernels", conv_q), ("fc_weights", fc_q)]:
        total = w.size
        pos   = (w ==  1).sum()
        neg   = (w == -1).sum()
        zero  = (w ==  0).sum()
        print(f"  {key}: +1={pos} ({pos/total*100:.1f}%), "
              f"-1={neg} ({neg/total*100:.1f}%), "
              f"0={zero} ({zero/total*100:.1f}%), "
              f"sparsity={zero/total*100:.1f}%")

    return {"conv_kernels": conv_q, "fc_weights": fc_q}


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────
def save_model(model, path="snps_twin_cnn.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")

def load_model(path="snps_twin_cnn.pth", device=DEVICE):
    model = SNPSTwinCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded ← {path}")
    return model

def save_quantized_weights(weights: dict, path="snps_weights.json"):
    """Save {-1,0,+1} weights as JSON for import into SNPS simulator."""
    serializable = {k: v.tolist() for k, v in weights.items()}
    with open(path, "w") as f:
        json.dump(serializable, f)
    print(f"Quantized weights saved → {path}")

def save_float_weights(weights: dict, path="snps_weights_float.npz"):
    """Save raw float weights for analysis or alternative ternarization."""
    np.savez(path, **weights)
    print(f"Float weights saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"Device:             {DEVICE}")
    print(f"Quantization range: Q={Q_RANGE}")
    print(f"Epochs:             {EPOCHS}")
    print(f"LR:                 {LR},  Batch: {BATCH_SIZE}")
    print(f"K threshold (FC):   {K_THRESHOLD}\n")

    # ── Data ──
    transform    = get_transforms(Q_RANGE)
    train_data   = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_data    = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Model ──
    model = SNPSTwinCNN()
    print(model)
    print(f"\nTotal trainable parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}\n")

    # ── Phase 1: Float training ──
    print("── Phase 1: Float training ──")
    history = train(model, train_loader, val_loader)

    # ── Save float checkpoint ──
    save_model(model, "snps_twin_cnn.pth")

    # ── Save float weights for analysis ──
    float_weights = {
        "conv_kernels": model.conv.weight.detach().cpu().numpy(),
        "fc_weights":   model.fc.weight.detach().cpu().numpy(),
    }
    save_float_weights(float_weights, "snps_weights_float.npz")

    # ── Phase 2: Ternarize ──
    print("\n── Phase 2: Ternarization ──")
    print(f"  Conv strategy : balanced per-filter (top-3 pos, top-3 neg per 3x3 kernel)")
    print(f"  FC strategy   : adaptive column-wise threshold (k={K_THRESHOLD})")
    ternary_weights = ternarize_weights(model, k=K_THRESHOLD)
    save_quantized_weights(ternary_weights, "snps_weights.json")

    print("\nDone. Files saved:")
    print("  snps_twin_cnn.pth        ← float model checkpoint")
    print("  snps_weights_float.npz   ← raw float weights (for analysis)")
    print("  snps_weights.json        ← ternary {-1,0,+1} weights → import in SNPS")

    return model, ternary_weights, history


if __name__ == "__main__":
    model, weights, history = main()
