#PRIMA: search for an easy CNN and try to recreate it -> https://pythonguides.com/simple-mnist-convnet-keras/
#LEGGERE DN-SNP: SNP-based lightweight deep network for CT image segmentation of COVID-19 o paper simili con CNN
#LEGGERE PER STDP: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full


# for cnn and kernels see https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148

import numpy as np
import time

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

from sps import handle_csv
from sps.digit_image import get_mnist_data
from sps.flower_image import get_flowers102_data
from sps.handle_csv import cnn_SNPS_csv, extend_csv, ensemble_csv
from sps.config import Config
from sps.med_image import get_med_mnist_data
from sps.snp_system import SNPSystem
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score




def launch_mnist_cnn():
    t=time.time()
    x_train, y_train, x_test, y_test = get_mnist_data()
    #example_direct(x_train, y_train, x_test, y_test) #compare with models baseline, launched directly on input images

    cnn_SNPS_csv() #create the csv for the SNPS
    svm, logreg = train_cnn(x_train, y_train)

    ensemble_accuracy = test_cnn(x_test, y_test, svm, logreg)
    handle_csv.save_results(ensemble_accuracy, time.time()-t)


def train_cnn(x_train, y_train):
    snps = SNPSystem(Config.TRAIN_SIZE, Config.TRAIN_SIZE + 5, True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)
    snps.spike_train = x_train
    snps.labels = y_train
    snps.start()

    #Support Vector Machine
    svm = LinearSVC(C=Config.SVM_C, max_iter=10000)
    svm.fit(snps.pooling_image.T, y_train)

    #Logistic Regression
    logreg = LogisticRegression(
        solver="lbfgs",
        max_iter=10000
    )
    logreg.fit(snps.pooling_image.T, y_train)

    return svm, logreg


def test_cnn(x_test, y_test, svm, logreg):

    compare_performance(x_test, y_test, svm, logreg) #for running and checking performance of other networks

    ensemble_pred = ensemble_and_test(x_test, svm.coef_, logreg.coef_, get_importance(svm.coef_), get_importance(logreg.coef_))
    ensemble_accuracy = np.mean(ensemble_pred == y_test)
    print("SNPS actual ensemble accuracy:", ensemble_accuracy)

    return ensemble_accuracy

def ensemble_and_test(x_test, svm_w, logreg_w, svm_imp, logreg_imp):
    snps = SNPSystem(Config.TEST_SIZE, Config.TEST_SIZE + 5, True)
    snps.spike_train = x_test
    svm_q = ternarize_matrix(svm_w.T)
    logreg_q = ternarize_matrix(logreg_w.T)
    extended_path = ensemble_csv(np.array(svm_q), np.array(logreg_q), svm_imp, logreg_imp)
    snps.load_neurons_from_csv(extended_path)
    snps.start()
    y_pred = np.argmax(snps.charge_map_prediction, axis=0)

    return y_pred



def get_importance(w):
    imp = compute_neuron_importance(w)
    if Config.DISCRETIZE_METHOD == 1: multipliers = discretize_percentile(imp)
    else: multipliers = discretize_proportional(imp) #Config.DISCRETIZE == 2
    return multipliers

def compute_neuron_importance(w):
    #Magnitude: neurons with larger overall weights across classes are considered more influential
    if Config.IMPORTANCE_METHOD == 1: imp = np.linalg.norm(w, axis=0)
    # Weight range: neurons whose weights vary more between classes are considered more discriminative
    else: imp = np.max(w, axis=0) - np.min(w, axis=0) #Config.IMPORTANCE_METHOD == 2
    imp = imp / max(imp.max(), 1e-8) #Normalize in range [0:1]
    return imp

def discretize_percentile(imp): #method 1 - percentile based importance
    p25 = np.percentile(imp, 25)   # first quartile
    p75 = np.percentile(imp, 75)   # third quartile
    multipliers = np.ones_like(imp)  # default multiplier = 1 (low importance)
    multipliers[imp > p75] = 3       # top 25% most important neurons
    multipliers[(imp > p25) & (imp <= p75)] = 2  # middle 50%
    multipliers[imp <= p25] = 1      # bottom 25%
    return multipliers.astype(int) # convert to integer

def discretize_proportional(imp): #method 2 - proportional based importance
    multipliers = 1 + np.round(imp * Config.DISC_RANGE)
    return multipliers.astype(int) # convert to integer



def ternarize_matrix(w):
    # matrix quantization for last layer of SNPS: Transform from real values to {-1,0,1}
    if Config.TERNARIZE_METHOD == 1: q = ternarize_percentile(w, Config.M_SPARSITY, Config.M_POSITIVE) # Percentile-based
    else: q = ternarize_threshold(w, Config.M_THRESHOLD) # Threshold-based
    return q

#https://www.emergentmind.com/topics/ternary-weight-networks-twns
#il link contiene info sulle reti ternarie TWN con pesi {-1,0,1}
def ternarize_percentile(w, p_zero, p_pos):
    """
    Ternary quantization {-1,0,1} using fixed percentiles per column.
    The smallest weights become -1, the largest become +1, the rest are 0.
    """
    p_neg = 1 - p_zero - p_pos
    w_q = np.zeros_like(w)
    n_rows, n_cols = w.shape

    for c in range(n_cols):
        col = w[:, c]
        sorted_idx = np.argsort(col)  # indices of sorted weights
        n_neg = int(p_neg * n_rows)
        n_zero = int(p_zero * n_rows)

        w_q[sorted_idx[:n_neg], c] = -1        # smallest weights
        w_q[sorted_idx[n_neg:n_neg+n_zero], c] = 0
        w_q[sorted_idx[n_neg+n_zero:], c] = 1  # largest weights

    return w_q

def ternarize_threshold(w, k):
    """
    Ternary quantization {-1,0,1} using a column-wise threshold.
    Weights larger than k * mean(|w|) become ±1, others become 0.
    """
    w_q = np.zeros_like(w)
    n_rows, n_cols = w.shape

    for c in range(n_cols):
        col = w[:, c]
        t = k * np.mean(np.abs(col))  # threshold for this column
        w_q[:, c] = np.where(col > t, 1, np.where(col < -t, -1, 0))  # ternary mapping

    return w_q



def example_direct(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    svm = LinearSVC(C=Config.SVM_C, max_iter=10000)
    svm.fit(x_train, y_train)

    #Logistic Regression
    logreg = LogisticRegression(
        solver="lbfgs",
        max_iter=10000
    )
    logreg.fit(x_train, y_train)

    print("svm direct accuracy", svm.score(x_test, y_test))
    print("logreg direct accuracy", logreg.score(x_test, y_test))


def compare_performance(x_test, y_test, svm, logreg):
    #-------------------------Testing the svm on SNPS-------------------------
    snps_svm_pred, _, svm_scores = extend_and_test(x_test,"svm", svm.coef_, None)
    snps_svm_accuracy = np.mean(snps_svm_pred == y_test)
    print("SNPS svm accuracy:", snps_svm_accuracy)

    #add importance
    snps_imp_svm_pred, features, svm_imp_scores = extend_and_test(x_test,"svm_imp", svm.coef_, get_importance(svm.coef_))
    snps_imp_svm_accuracy = np.mean(snps_imp_svm_pred == y_test)
    print("SNPS svm imp accuracy:", snps_imp_svm_accuracy)

    features_int_pos = np.maximum(features // 4, 0) #extract feature from images

    #real weights SVM
    raw_svm_accuracy = svm.score(features_int_pos.T, y_test)
    print("real weights svm accuracy:", raw_svm_accuracy)

    #-------------------------Testing the logreg on SNPS-------------------------
    snps_lr_pred, _, lr_scores = extend_and_test(x_test,"lr", logreg.coef_, None)
    snps_lr_accuracy = np.mean(snps_lr_pred == y_test)
    print("SNPS logreg accuracy:", snps_lr_accuracy)

    #add importance
    snps_imp_lr_pred, _, lr_imp_scores = extend_and_test(x_test,"lr_imp", logreg.coef_, get_importance(logreg.coef_))
    snps_imp_lr_accuracy = np.mean(snps_imp_lr_pred == y_test)
    print("SNPS logreg imp accuracy:", snps_imp_lr_accuracy)

    #real weights LOGREG
    raw_lr_accuracy = logreg.score(features_int_pos.T, y_test)
    print("real weights logreg accuracy:", raw_lr_accuracy)

    #------------------combined charge------------
    # sum charge of the models without importance
    sum_pred = svm_scores.T + lr_scores.T
    sum_labels = np.argmax(sum_pred, axis=0)
    ens_accuracy = np.mean(sum_labels == y_test)
    print("SNPS ensemble accuracy:", ens_accuracy)

    #------------------combined IMP charge------------
    # sum charge of the models with importance (same as the SNPS with ensemble)
    sum_imp_pred = svm_imp_scores.T + lr_imp_scores.T
    sum_imp_labels = np.argmax(sum_imp_pred, axis=0)
    ens_imp_accuracy = np.mean(sum_imp_labels == y_test)
    print("SNPS imp ensemble accuracy:", ens_imp_accuracy)


def extend_and_test(x_test, method, w, multipliers):
    snps = SNPSystem(Config.TEST_SIZE, Config.TEST_SIZE + 5, True)
    snps.spike_train = x_test
    q = ternarize_matrix(w.T)
    extended_path = extend_csv("csv/" + Config.CSV_NAME, np.array(q), method, multipliers)
    snps.load_neurons_from_csv(extended_path)
    snps.start()
    y_pred = np.argmax(snps.charge_map_prediction, axis=0)

    return y_pred, snps.pooling_image, snps.charge_map_prediction.T
