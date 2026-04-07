#PRIMA: search for an easy CNN and try to recreate it -> https://pythonguides.com/simple-mnist-convnet-keras/
#LEGGERE DN-SNP: SNP-based lightweight deep network for CT image segmentation of COVID-19 o paper simili con CNN
#LEGGERE PER STDP: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full


# for cnn and kernels see https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148

import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

from sps import handle_csv
from sps.digit_image import get_mnist_data
from sps.handle_csv import cnn_SNPS_csv, extend_csv
from sps.config import Config
from sps.snp_system import SNPSystem
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

def test_launch_mnist_cnn():
    x_train, y_train, x_test, y_test = get_mnist_data()
    cnn_SNPS_csv() #use only if the csv was changed

def launch_mnist_cnn():
    t=time.time()
    x_train, y_train, x_test, y_test = get_mnist_data()
    cnn_SNPS_csv() #use only if the csv was changed
    svm, logreg = train_cnn(x_train, y_train)
    train_time = time.time() - t

    #test phase
    t=time.time()
    svm_accuracy, lr_accuracy, ens_accuracy, ens_imp_accuracy, raw_svm_accuracy, raw_lr_accuracy = test_cnn(x_test, y_test, svm, logreg)
    handle_csv.save_results(svm_accuracy, lr_accuracy, ens_accuracy, ens_imp_accuracy, raw_svm_accuracy, raw_lr_accuracy, time.time()-t+train_time)


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
        max_iter=1000
    )
    logreg.fit(snps.pooling_image.T, y_train)

    return svm, logreg #snps.model.get_synapses()

def print_results(y_test, pred, scores):
    y_test_bin = label_binarize(y_test, classes=np.arange(10)) #ROC curve
    roc_auc = roc_auc_score(
        y_test_bin,
        scores,
        multi_class="ovr",
        average="macro"
    )
    print("ROC AUC:", roc_auc)
    print("F1 macro:", f1_score(y_test, pred, average="macro"))
    print("F1 weighted:", f1_score(y_test, pred, average="weighted"))
    print(classification_report(y_test, pred))

def test_cnn(x_test, y_test, svm, logreg):

    #-------------------------Testing the svm on SNPS-------------------------
    snps_svm_pred, _, svm_scores = extend_and_test(x_test,"svm", svm.coef_, None)
    snps_svm_accuracy = np.mean(snps_svm_pred == y_test)
    print("SNPS svm not_imp accuracy:", snps_svm_accuracy)
    #print_results(y_test, svm_scores)


    snps_imp_svm_pred, features, svm_imp_scores = extend_and_test(x_test,"svm_imp", svm.coef_, get_importance(svm.coef_))
    snps_imp_svm_accuracy = np.mean(snps_imp_svm_pred == y_test)
    print("SNPS svm accuracy:", snps_imp_svm_accuracy)
    #print_results(y_test, snps_imp_svm_pred, svm_imp_scores)


    features_int_pos = np.maximum(features // 4, 0)

    #SVM
    #svm_pred = svm.predict(features_int_pos.T)
    raw_svm_accuracy = svm.score(features_int_pos.T, y_test)
    print("raw svm accuracy:", raw_svm_accuracy)
    #print_results(y_test, svm_pred)


    #-------------------------Testing the logreg on SNPS-------------------------
    snps_lr_pred, _, lr_scores = extend_and_test(x_test,"lr", logreg.coef_, None)
    lr_accuracy = np.mean(snps_lr_pred == y_test)
    print("SNPS logreg not_imp accuracy:", lr_accuracy)
    #print_results(y_test, snps_lr_pred)


    snps_imp_lr_pred, _, lr_imp_scores = extend_and_test(x_test,"lr_imp", logreg.coef_, get_importance(logreg.coef_))
    snps_imp_lr_accuracy = np.mean(snps_imp_lr_pred == y_test)
    print("SNPS logreg accuracy:", snps_imp_lr_accuracy)
    #print_results(y_test, snps_imp_lr_pred, lr_scores)


    #LOGREG
    #lr_pred = logreg.predict(features_int_pos.T)
    raw_lr_accuracy = logreg.score(features_int_pos.T, y_test)
    print("raw logreg accuracy:", raw_lr_accuracy)
    #print_results(y_test, lr_pred)

    #------------------combined charge------------ #TODO add new layer, split layer 3
    # somma delle cariche dei due modelli
    sum_pred = svm_scores.T + lr_scores.T
    sum_labels = np.argmax(sum_pred, axis=0)
    ens_accuracy = np.mean(sum_labels == y_test)
    print("-SNPS ensemble accuracy:", ens_accuracy)

    #------------------combined IMP charge------------
    # somma delle cariche dei due modelli
    sum_imp_pred = svm_imp_scores.T + lr_imp_scores.T
    sum_imp_labels = np.argmax(sum_imp_pred, axis=0)
    ens_imp_accuracy = np.mean(sum_imp_labels == y_test)
    print("-SNPS imp ensemble accuracy:", ens_imp_accuracy)


    return snps_imp_svm_accuracy, snps_imp_lr_accuracy, ens_accuracy, ens_imp_accuracy, raw_svm_accuracy, raw_lr_accuracy



def extend_and_test(x_test, method, w, multipliers):
    snps = SNPSystem(Config.TEST_SIZE, Config.TEST_SIZE + 5, True)
    snps.spike_train = x_test
    q = quantize_matrix(w.T)
    extended_path = extend_csv("csv/" + Config.CSV_NAME, np.array(q), method, multipliers)
    snps.load_neurons_from_csv(extended_path)
    snps.start()
    y_pred = np.argmax(snps.charge_map_prediction, axis=0)

    return y_pred, snps.pooling_image, snps.charge_map_prediction.T

def get_importance(w):
    alpha = compute_neuron_importance(w)
    if Config.DISCRETIZE_METHOD == 1: multipliers = discretize_percentile(alpha)
    else: multipliers = discretize_proportional(alpha) #Config.DISCRETIZE == 2

    return multipliers

def compute_neuron_importance(w):
    #Magnitude: neurons with larger overall weights across classes are considered more influential
    if Config.ALPHA_METHOD == 1: alpha = np.linalg.norm(w, axis=0) #alpha 1
    # Weight range: neurons whose weights vary more between classes are considered more discriminative
    else: alpha = np.max(w, axis=0) - np.min(w, axis=0) #Config.ALPHA_METHOD == 2
    alpha = alpha / max(alpha.max(), 1e-8) #Normalize in range [0:1]
    return alpha


def discretize_percentile(alpha): #method 1 - percentile based importance
    p25 = np.percentile(alpha, 25)   # first quartile
    p75 = np.percentile(alpha, 75)   # third quartile
    multipliers = np.ones_like(alpha)  # default multiplier = 1 (low importance)

    multipliers[alpha > p75] = 3       # top 25% most important neurons
    multipliers[(alpha > p25) & (alpha <= p75)] = 2  # middle 50%
    multipliers[alpha <= p25] = 1      # bottom 25%
    return multipliers.astype(int) # convert to integer

def discretize_proportional(alpha): #method 2 - proportional based importance
    multipliers = 1 + np.round(alpha * Config.DISC_RANGE)
    return int(multipliers) # convert to integer


def quantize_matrix(w):
    # matrix quantization for last layer of SNPS: Transform from real values to {-1,0,1}
    if Config.QUANTIZE_METHOD == 1: q = quantize_percentile(w, Config.M_SPARSITY, Config.M_POSITIVE) # Percentile-based
    else: q = quantize_threshold(w, Config.M_THRESHOLD) # Threshold-based
    return q


#https://www.emergentmind.com/topics/ternary-weight-networks-twns
#il link contiene info sulle reti ternarie TWN con pesi {-1,0,1}
def quantize_percentile(w, p_zero, p_pos):
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

def quantize_threshold(w, k):
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



"""
!!!ATTENZIONE: FINO A QUI HO SBAGLIATO E AVEVO svm", svm.coef_, get_importance(logreg.coef_))!
quindi ignorare prima del 2026-03-06 14:10:52
------------------------
uno dei primi ensemble ha ottenuto 94.6%
{"train size": 3000, "test size": 500, "q range": 10, "svm c": 1.0, "quantize method": 3, "alpha method": 2, "discretize method": 1, "discretization range": 2, "matrix sparsity": 0.5, "matrix positive": 0.25, "matrix threshold": 0.5, "database": "digit", "kernel number": 8}
{"SVM accuracy": 0.92, "LR accuracy": 0, "time": 326.88240218162537}

TRAIN: 3000
SNPS svm not_imp accuracy: 0.92
SNPS svm accuracy: 0.92
raw svm accuracy: 0.95
SNPS logreg not_imp accuracy: 0.9
SNPS logreg accuracy: 0.922
raw logreg accuracy: 0.958

SNPS ensemble accuracy: 0.928
SNPS imp ensemble accuracy: 0.946
-------------------------------
---simulazione del 2026-03-07 17:59:44 - SPARSITY: 0.8 - QRANGE: 10
{"train size": 5000, "test size": 1000, "q range": 10, "svm c": 1.0, "quantize method": 1, "alpha method": 2, "discretize method": 1, "discretization range": 2, "matrix sparsity": 0.8, "matrix positive": 0.1, "matrix threshold": 0.5, "database": "digit", "kernel number": 8}
{"SVM accuracy": 0.924, "LR accuracy": 0.922, "ens accuracy": 0.936, "ens imp accuracy": 0.941, "time": 643.7142617702484}
------------------------
---Fisso QRANGE a 10, SPARSITY: 0.8, e vario train size esponenzialmente per vedere se davvero calano le performance:
la migliore l'ho ottenuta con TRAIN 4000 e TEST 1000 e ha 93.8%:
{"train size": 4000, "test size": 1000, "q range": 10, "svm c": 1.0, "quantize method": 1, "alpha method": 2, "discretize method": 1, "discretization range": 2, "matrix sparsity": 0.8, "matrix positive": 0.1, "matrix threshold": 0.5, "database": "digit", "kernel number": 8}
{"SVM accuracy": 0.906, "LR accuracy": 0.921, "ens accuracy": 0.924, "ens imp accuracy": 0.938, "raw svm accuracy": 0.947, "raw lr accuracy": 0.955, "time": 744.9951825141907}
---------------------------------------
---prove con quantizzazione a metodo 2 e vari k (M_THRESHOLD):
il migliore l'ho ottenuto con k:1.5 e ha 95%:
{"train size": 5000, "test size": 1000, "q range": 5, "svm c": 1.0, "quantize method": 2, "alpha method": 2, "discretize method": 1, "discretization range": 2, "matrix sparsity": 0.8, "matrix positive": 0.1, "matrix threshold": 1.5, "database": "digit", "kernel number": 8}
{"SVM accuracy": 0.911, "LR accuracy": 0.936, "ens accuracy": 0.941, "ens imp accuracy": 0.949, "raw svm accuracy": 0.944, "raw lr accuracy": 0.958, "time": 335.30409836769104} 

------------------------------
---ho messo alpha method = 1, mantenuto il miglior risultato precedente e ottenuto 94.9%:
{"train size": 5000, "test size": 1000, "q range": 5, "svm c": 1.0, "quantize method": 2, "alpha method": 1, "discretize method": 1, "discretization range": 2, "matrix sparsity": 0.8, "matrix positive": 0.1, "matrix threshold": 1.5, "database": "digit", "kernel number": 8}
{"SVM accuracy": 0.913, "LR accuracy": 0.937, "ens accuracy": 0.941, "ens imp accuracy": 0.949, "raw svm accuracy": 0.944, "raw lr accuracy": 0.958, "time": 320.3761622905731}
----------------------------


    """
