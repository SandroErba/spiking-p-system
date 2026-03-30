import medmnist
import numpy as np
from matplotlib import pyplot as plt
from medmnist import INFO
from sps.config import Config


def get_med_mnist_data(data_name=None, rgb_to_gray_method="pca"):
    """Return MedMNIST data as x_train, y_train, x_test, y_test (digit-style preprocessing)."""
    data_name = data_name or Config.DATABASE
    if data_name not in INFO:
        available = ", ".join(sorted(INFO.keys()))
        raise ValueError(f"Unknown MedMNIST dataset '{data_name}'. Available: {available}")

    info = INFO[data_name]
    data_class = getattr(medmnist, info['python_class'])
    train_dataset = data_class(split='train', download=True)
    test_dataset = data_class(split='test', download=True)

    train_imgs = train_dataset.imgs[:Config.TRAIN_SIZE]
    test_imgs = test_dataset.imgs[:Config.TEST_SIZE]

    train_data = to_grayscale_batch(train_imgs, rgb_to_gray_method)
    test_data = to_grayscale_batch(test_imgs, rgb_to_gray_method)

    train_label = train_dataset.labels[:Config.TRAIN_SIZE].flatten()
    test_label = test_dataset.labels[:Config.TEST_SIZE].flatten()

    # Same quantization strategy used in digit_image.py.
    train_q = ((train_data.astype(np.float32) * Config.Q_RANGE) // 256).astype(np.uint8)
    test_q = ((test_data.astype(np.float32) * Config.Q_RANGE) // 256).astype(np.uint8)
    if Config.INVERT:
        train_q = Config.Q_RANGE - train_q
        test_q = Config.Q_RANGE - test_q

    return train_q, train_label, test_q, test_label


def to_grayscale_batch(imgs, method="pca"):
    # Already grayscale: (N, H, W)
    if imgs.ndim == 3:
        return imgs

    # Single-channel grayscale: (N, H, W, 1)
    if imgs.ndim == 4 and imgs.shape[-1] == 1:
        return imgs[..., 0]

    # RGB: (N, H, W, 3)
    if imgs.ndim == 4 and imgs.shape[-1] == 3:
        if method == "luminance":
            return rgb_to_gray_luminance(imgs)
        if method == "pca":
            return rgb_to_gray_pca(imgs)
        raise ValueError("Unknown rgb_to_gray_method. Use 'luminance' or 'pca'.")

    raise ValueError(f"Unsupported image tensor shape: {imgs.shape}")


def rgb_to_gray_luminance(imgs):
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.tensordot(imgs.astype(np.float32), weights, axes=([-1], [0]))
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def rgb_to_gray_pca(imgs):
    out = np.empty(imgs.shape[:3], dtype=np.uint8)
    for i in range(imgs.shape[0]):
        rgb = imgs[i].astype(np.float32)
        flat = rgb.reshape(-1, 3)
        mean = flat.mean(axis=0, keepdims=True)
        centered = flat - mean
        cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_vec = eigvecs[:, np.argmax(eigvals)]

        # Keep orientation stable.
        if principal_vec.sum() < 0:
            principal_vec = -principal_vec

        projected = centered @ principal_vec
        projected = projected.reshape(rgb.shape[:2])

        min_val = float(np.min(projected))
        max_val = float(np.max(projected))
        if max_val > min_val:
            norm = (projected - min_val) / (max_val - min_val)
        else:
            norm = np.zeros_like(projected, dtype=np.float32)

        out[i] = np.clip(np.rint(norm * 255), 0, 255).astype(np.uint8)

    return out





def show_samples(x, y, train=False):
    x_np = np.asarray(x)
    y_np = np.asarray(y).ravel()

    nrows = 3
    ncols = 10
    if train and Config.TRAIN_SIZE < 30:
        nrows = 2
        ncols = max(1, int((Config.TRAIN_SIZE - 1) / 2))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3))

    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            if idx >= len(x_np):
                axes[row, col].axis("off")
                continue

            image = x_np[idx].reshape((Config.IMG_SHAPE, Config.IMG_SHAPE))
            ax = axes[row, col]
            ax.imshow(image, cmap="gray_r", interpolation="nearest")
            ax.set_title(f"Label: {y_np[idx]}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()
