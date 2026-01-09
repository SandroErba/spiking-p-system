#TODO try to test quantized SNPS with this
from torchvision import transforms
from torchvision.datasets import Flowers102
import numpy as np
from sps.config import Config
from sps.med_image import quantize_rgb_image, binarize_rgb_image, show_quantized_image


def get_flowers102_data():
    Config.INVERT = False
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # IMPORTANT: uniform size
    ])

    train_dataset = Flowers102(
        root="./data",
        split="train",
        transform=transform,
        download=True
    )

    test_dataset = Flowers102(
        root="./data",
        split="test",
        transform=transform,
        download=True
    )

    return (
        process_flowers102_dataset(train_dataset, Config.TRAIN_SIZE),
        process_flowers102_dataset(test_dataset, Config.TEST_SIZE),
    )


def process_flowers102_dataset(dataset, count):
    imgs = []
    red_channel = []
    green_channel = []
    blue_channel = []
    labels = []

    for i in range(count):
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
        imgs.append(img)

    #show_quantized_image(imgs[0], red_channel[0], green_channel[0], blue_channel[0]) #Show first image
    #show_quantized_image(img[1], red_channel[1], green_channel[1], blue_channel[1]) #Show second image
    #show_quantized_image(img[2], red_channel[2], green_channel[2], blue_channel[2]) #Show third image

    return (
        np.array(red_channel),
        np.array(green_channel),
        np.array(blue_channel),
        np.array(labels),
    )