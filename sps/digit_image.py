from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import numpy as np
from sps.config import Config
from sps.handle_csv import quantized_SNPS_csv
from sps.med_mnist import syn_train_SNPS, compute_SNPS, combined_ranking_score


def get_digit_data():
    optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)
    x = optical_recognition_of_handwritten_digits.data.features
    y = optical_recognition_of_handwritten_digits.data.targets

    #show_digit(x, y) #TODO for showing the input images

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        train_size=Config.TRAIN_SIZE,
        test_size=Config.TEST_SIZE,
        shuffle=True,
        random_state=42
    )
    x_train = x_train.to_numpy(dtype=np.int16)
    x_test  = x_test.to_numpy(dtype=np.int16)
    x_train_q = np.round(x_train * Config.Q_RANGE / 16).astype(np.uint8)
    x_test_q = np.round(x_test * Config.Q_RANGE / 16).astype(np.uint8)
    y_train_np = y_train.to_numpy().ravel()
    y_test_np = y_test.to_numpy().ravel()

    #show_digit(x_train_q, y_train) #TODO for showing the quantized images
    #print("x_train", x_train[0])
    #print("x_train_q", x_train_q[0])

    return x_train_q, y_train_np, x_test_q, y_test_np


def launch_gray_SNPS():
    """Manage grayscale quantized SN P system"""
    train_data, train_labels, test_data, test_labels = get_digit_data()

    quantized_SNPS_csv()                       # prepare CSV for this color
    syn_train_SNPS(train_data, train_labels)   # prune + inhibit
    predictions = compute_SNPS(test_data)      # test

    #print("Predictions shape:", predictions.shape) #TODO one class, the 8 (ninth class) has high charge and get all the prob
    #TODO !print the predicted labels to check how the model is classifing!

    #for i in range(10):
    #    print("red Predictions :", i,  predictions[i]) #TODO NOW: charge is too similar for every class

    combined_ranking_score(predictions, predictions, predictions, test_labels)


def show_digit(x, y):
    x_np = x
    if not isinstance(x, np.ndarray):
        x_np = x.to_numpy()
    y_np = y.to_numpy().ravel()  # make labels 1D

    nrows = 3
    ncols = 10

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3))

    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            image = x_np[idx].reshape(8, 8)

            ax = axes[row, col]
            ax.imshow(image, cmap="gray_r", interpolation="nearest")
            ax.set_title(f"Label: {y_np[idx]}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()