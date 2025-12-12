import matplotlib.pyplot as plt
import medmnist
import numpy as np
from medmnist import INFO
from sps.config import Config

def get_mnist_data(dataName): # download the database
    info = INFO[dataName]
    data_class = getattr(medmnist, info['python_class'])
    train_dataset = data_class(split='train', download=True)
    test_dataset = data_class(split='test', download=True)

    return (
        (process_dataset(train_dataset, Config.TRAIN_SIZE)),
        (process_dataset(test_dataset, Config.TEST_SIZE))
    )


def process_dataset(dataset, count): # flatten and split among color channels
    imgs = dataset.imgs[:count]
    labels = dataset.labels[:count].flatten()
    red_channel = []
    green_channel = []
    blue_channel = []

    for img in imgs:
        if Config.QUANTIZATION:
            ch_r, ch_g, ch_b = quantize_rgb_image(img)
        else:
            ch_r, ch_g, ch_b = binarize_rgb_image(img)
        red_channel.append(ch_r)
        green_channel.append(ch_g)
        blue_channel.append(ch_b)

    #show_quantized_image(dataset.imgs[0], red_channel[0], green_channel[0], blue_channel[0]) #Show first image
<<<<<<< HEAD
=======
    #show_quantized_image(dataset.imgs[1], red_channel[1], green_channel[1], blue_channel[1]) #Show first image
    #show_quantized_image(dataset.imgs[2], red_channel[2], green_channel[2], blue_channel[2]) #Show first image
>>>>>>> 2c97215ec8e5ed5131013ed1d66014bbead5d477

    return (
        np.array(red_channel),
        np.array(green_channel),
        np.array(blue_channel),
        labels
    )

def binarize_rgb_image(img_rgb): # binarize for create the input array
    binary_channels = 1 - (img_rgb > int(Config.THRESHOLD)).astype(int) # From [0,255] to [0,1]
    downsampled = []
    for c in range(3):
        ch = binary_channels[:, :, c]
        downsampled.append(ch)
    return downsampled  # List of 3 arrays of 784 bit

def quantize_rgb_image(img_rgb):
    # Divide into 5 ranges: 0, 1–63, 64–127, 128–191, 192–255 TODO can be done from 0-63 -> 1 or 0-63 -> 0 (max 3 spike)
    quantized = np.ceil(img_rgb.astype(float) / 64).astype(int)
    quantized[img_rgb == 0] = 0
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


<<<<<<< HEAD
=======
#for output an image
def show_images(output_array):
    images = np.asarray(output_array)
    num_images = min(images.shape[1], Config.TRAIN_SIZE)
    cols = min(num_images, 5)
    rows = int(np.ceil(num_images / cols))
    plt.figure(figsize=(2.5 * cols, 2.5 * rows))

    for i in range(num_images):
        img = images[:, i].reshape((Config.SEGMENTED_SHAPE, Config.SEGMENTED_SHAPE))
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Image {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()



>>>>>>> 2c97215ec8e5ed5131013ed1d66014bbead5d477
# Debug only
def show_quantized_image(original_img, q_r, q_g, q_b):
    """
    original_img: original RGB image (H,W,3)
    q_r, q_g, q_b: quantized channels (values 0–4)
    """

    # Convert quantized levels back to 0–255 for visualization
    # Level 0 = 0
    # Level 1–4 spread across intensity range
    def expand_channel(ch):
        return (ch / 4.0 * 255).astype(np.uint8)

    r = expand_channel(q_r)
    g = expand_channel(q_g)
    b = expand_channel(q_b)

    reconstructed = np.stack([r, g, b], axis=-1)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(original_img)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Quantized R (0–4)")
    plt.imshow(q_r, cmap="gray", vmin=0, vmax=4)
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Quantized G (0–4)")
    plt.imshow(q_g, cmap="gray", vmin=0, vmax=4)
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("Quantized B (0–4)")
    plt.imshow(q_b, cmap="gray", vmin=0, vmax=4)
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Reconstructed (using 0–255 remap)")
    plt.imshow(reconstructed)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
