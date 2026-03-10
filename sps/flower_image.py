from torchvision import transforms
from torchvision.datasets import Flowers102
import numpy as np
from sps.config import Config

# TODO class: what i should do with this database?
def get_flowers102_data():
    Config.INVERT = False
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    train_dataset = Flowers102(
        root="./data",
        split="train",
        transform=transform,
        download=True,
    )

    test_dataset = Flowers102(
        root="./data",
        split="test",
        transform=transform,
        download=True,
    )

    train_x, train_y = process_flowers102_dataset(train_dataset, Config.TRAIN_SIZE, "train")
    test_x, test_y = process_flowers102_dataset(test_dataset, Config.TEST_SIZE, "test")

    # Use only classes observed in train, then remap labels to [0..K-1].
    train_classes = np.unique(train_y)
    if train_classes.size < 2:
        raise ValueError("Flower train split has less than 2 classes. Increase TRAIN_SIZE.")

    class_to_idx = {int(c): i for i, c in enumerate(train_classes.tolist())}
    train_y = np.array([class_to_idx[int(c)] for c in train_y], dtype=np.int32)

    # Drop test samples from unseen classes to keep classifier/output dimensions aligned.
    test_mask = np.isin(test_y, train_classes)
    if not np.all(test_mask):
        removed = int((~test_mask).sum())
        print(f"Flower test filtering: removed {removed} samples from unseen classes")
    test_x = test_x[test_mask]
    test_y = np.array([class_to_idx[int(c)] for c in test_y[test_mask]], dtype=np.int32)

    Config.CLASSES = int(train_classes.size)
    Config.TRAIN_SIZE = int(train_x.shape[0])
    Config.TEST_SIZE = int(test_x.shape[0])
    print(f"Flower effective classes: {Config.CLASSES} | train={Config.TRAIN_SIZE} test={Config.TEST_SIZE}")

    return train_x, train_y, test_x, test_y


def process_flowers102_dataset(dataset, count, split_name="data"):
    red_channel = []
    green_channel = []
    blue_channel = []
    labels = []

    limit = min(count, len(dataset))
    print(f"Flower preprocessing ({split_name}): 0/{limit}")
    for i in range(limit):
        img, label = dataset[i]     # PIL image + int label
        img = np.array(img)         # (H, W, 3), uint8


        if Config.QUANTIZATION:
            ch_r, ch_g, ch_b = quantize_rgb_image(img)
        else:
            ch_r, ch_g, ch_b = binarize_rgb_image(img)

        red_channel.append(ch_r)
        green_channel.append(ch_g)
        blue_channel.append(ch_b)
        labels.append(label)

        if (i + 1) % 20 == 0 or (i + 1) == limit:
            print(f"Flower preprocessing ({split_name}): {i + 1}/{limit}")

    #show_quantized_image(img, red_channel[0], green_channel[0], blue_channel[0]) #Show first image
    #show_quantized_image(img[1], red_channel[1], green_channel[1], blue_channel[1]) #Show second image
    #show_quantized_image(img[2], red_channel[2], green_channel[2], blue_channel[2]) #Show third image

    return (
        merge_rgb_channels_to_grayscale(red_channel, green_channel, blue_channel),  # (N, H, W)
        np.array(labels)  # (N,)        
    )


def binarize_rgb_image(img_rgb):
    # From [0,255] to [0,1] on each channel.
    binary_channels = 1 - (img_rgb > int(Config.THRESHOLD)).astype(int)
    downsampled = []
    for c in range(3):
        downsampled.append(binary_channels[:, :, c])
    return downsampled


def quantize_rgb_image(img_rgb):
    threshold_fac = 256 / Config.Q_RANGE
    quantized = np.round(img_rgb.astype(float) / threshold_fac).astype(int)
    inverted = np.where(
        quantized == 0,
        0,
        Config.Q_RANGE + 1 - quantized
    )
    channels = []
    if Config.INVERT:
        for c in range(3):
            channels.append(inverted[:, :, c])
    else:
        for c in range(3):
            channels.append(quantized[:, :, c])
    return channels


def merge_rgb_channels_to_grayscale(red_channel, green_channel, blue_channel):
    red = np.asarray(red_channel)
    green = np.asarray(green_channel)
    blue = np.asarray(blue_channel)

    rgb = np.stack([red.astype(np.float32), green.astype(np.float32), blue.astype(np.float32)], axis=-1)
    flat = rgb.reshape(-1, 3)

    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean

    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_vec = eigvecs[:, np.argmax(eigvals)]

    if principal_vec.sum() < 0:
        principal_vec = -principal_vec

    pca_values = centered @ principal_vec
    pca_gray = pca_values.reshape(red.shape)

    min_val = float(np.min(pca_gray))
    max_val = float(np.max(pca_gray))
    if max_val > min_val:
        gray = (pca_gray - min_val) / (max_val - min_val)
    else:
        gray = np.zeros_like(pca_gray, dtype=np.float32)

    if np.issubdtype(red.dtype, np.integer):
        info = np.iinfo(red.dtype)
        scaled = gray * info.max
        return np.clip(np.rint(scaled), info.min, info.max).astype(red.dtype)

    return gray.astype(np.float32)