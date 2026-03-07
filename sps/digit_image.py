from keras.src.datasets import mnist
from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import numpy as np
from sps.config import Config
from sps.handle_csv import quantized_SNPS_csv
from sps.med_mnist import syn_train_SNPS, compute_SNPS, combined_ranking_score


#from tensorflow.keras.datasets import mnist
def get_28_digit_data():
    (train_data, train_label), (test_data, test_label) = mnist.load_data()
    train_data = train_data[:Config.TRAIN_SIZE]
    train_label = train_label[:Config.TRAIN_SIZE]
    test_data = test_data[:Config.TEST_SIZE]
    test_label = test_label[:Config.TEST_SIZE]
    train_q = ((train_data.astype(np.float32) * Config.Q_RANGE) // 256).astype(np.uint8)
    test_q = ((test_data.astype(np.float32) * Config.Q_RANGE) // 256).astype(np.uint8)
    if Config.INVERT:
        train_q = Config.Q_RANGE - train_q
        test_q = Config.Q_RANGE - test_q
    #print(train_q[0])
    #plt.imshow(train_q[0])
    #plt.show()
    return train_q, train_label, test_q, test_label


def get_digit_data():
    optical_recognition_of_handwritten_digits = fetch_ucirepo(id=80)
    x = optical_recognition_of_handwritten_digits.data.features
    y = optical_recognition_of_handwritten_digits.data.targets

    #show_digit(x, y) #for showing the input images

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

    #show_digit(x_train_q, y_train, True) #for showing the quantized images
    #print("x_train", x_train[0])
    #print("x_train_q", x_train_q[0])
    if Config.INVERT:
        x_train_q = Config.Q_RANGE - x_train_q
        x_test_q = Config.Q_RANGE - x_test_q

    return x_train_q, y_train_np, x_test_q, y_test_np


def launch_gray_SNPS():
    """Manage grayscale quantized SN P system"""
    train_data, train_labels, test_data, test_labels = get_digit_data()

    from sps.handle_csv import cnn_SNPS_csv # Aggiungi questo import in alto al file o qui

    print("CNN")
    cnn_SNPS_csv()                             # USIAMO LA CNN INVECE DELLA RETE BASE
    syn_train_SNPS(train_data, train_labels)   # prune + inhibit
    predictions = compute_SNPS(test_data)      # test

    #print("Predictions shape:", predictions.shape)
    #TODO !print the predicted labels to check how the model is classifing!
    # one class, the 8 (ninth class) has high charge and get all the prob

    #for i in range(10):
    #    print("red Predictions :", i,  predictions[i]) #TODO NOW: charge is too similar for every class

    combined_ranking_score(predictions, predictions, predictions, test_labels)


def show_digit(x, y, train = False):
    x_np = x
    if not isinstance(x, np.ndarray):
        x_np = x.to_numpy()
    y_np = y.to_numpy().ravel()  # make labels 1D

    nrows = 3
    ncols = 10
    if train and Config.TRAIN_SIZE < 30:
        nrows = 2
        ncols = int((Config.TRAIN_SIZE - 1) / 2)

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