#PRIMA: search for an easy CNN and try to recreate it -> https://pythonguides.com/simple-mnist-convnet-keras/
#LEGGERE DN-SNP: SNP-based lightweight deep network for CT image segmentation of COVID-19 o paper simili con CNN
#LEGGERE PER STDP: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full


# for cnn and kernels see https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
#guardare il lavoro di Iris Ermini, analizza immagini grandi binarizzandole

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



def launch_mnist_cnn():
    t=time.time()
    x_train, y_train, x_test, y_test = get_mnist_data()
    cnn_SNPS_csv() #use only if the csv was changed
    svm, logreg = train_cnn(x_train, y_train)
    train_time = time.time() - t

    t=time.time()
    svm_accuracy, lr_accuracy = test_cnn(x_test, y_test, svm, logreg)
    handle_csv.save_results(svm_accuracy, lr_accuracy, time.time()-t+train_time)






    return -1, -1


def train_cnn(x_train, y_train):
    snps = SNPSystem(Config.TRAIN_SIZE, Config.TRAIN_SIZE + 5, True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)
    snps.spike_train = x_train
    snps.labels = y_train
    snps.start()


    #SVM
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
    #print("F1 macro:", f1_score(y_test, pred, average="macro"))
    #print("F1 weighted:", f1_score(y_test, pred, average="weighted"))
    #print(classification_report(y_test, pred))

def test_cnn(x_test, y_test, svm, logreg):

    #-------------------------Testing the svm on SNPS-------------------------
    #snps_svm_pred, features = extend_and_test(x_test,"svm", svm.coef_, None)
    #snps_svm_accuracy = np.mean(snps_svm_pred == y_test)
    #print("------SNPS svm accuracy:", snps_svm_accuracy)
    #print_results(y_test, snps_svm_pred)


    snps_imp_svm_pred, features, svm_scores = extend_and_test(x_test,"svm", svm.coef_, get_importance(svm.coef_))
    snps_imp_svm_accuracy = np.mean(snps_imp_svm_pred == y_test)
    print("SNPS svm accuracy:", snps_imp_svm_accuracy)
    #print_results(y_test, snps_imp_svm_pred, svm_scores)


    features_int_pos = np.maximum(features // 4, 0)

    #SVM
    svm_pred = svm.predict(features_int_pos.T)
    print("raw svm accuracy:", svm.score(features_int_pos.T, y_test))
    #print_results(y_test, svm_pred)


    #-------------------------Testing the logreg on SNPS-------------------------
    #snps_lr_pred, _ = extend_and_test(x_test,"lr", logreg.coef_, None)
    #lr_accuracy = np.mean(snps_lr_pred == y_test)
    #print("------SNPS logreg accuracy:", lr_accuracy)
    #print_results(y_test, snps_lr_pred)


    snps_imp_lr_pred, _, lr_scores = extend_and_test(x_test,"lr", logreg.coef_, get_importance(logreg.coef_))
    snps_imp_lr_accuracy = np.mean(snps_imp_lr_pred == y_test)
    print("SNPS logreg accuracy:", snps_imp_lr_accuracy)
    #print_results(y_test, snps_imp_lr_pred, lr_scores)


    #LOGREG
    lr_pred = logreg.predict(features_int_pos.T)
    print("raw logreg accuracy:", logreg.score(features_int_pos.T, y_test))
    #print_results(y_test, lr_pred)

    return snps_imp_svm_accuracy, snps_imp_lr_accuracy



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
    if Config.ALPHA_METHOD == 1: alpha = np.linalg.norm(w, axis=0) #alpha 1 - magnitude
    else: alpha = np.max(w, axis=0) - np.min(w, axis=0) #Config.ALPHA == 2
    alpha = alpha / (alpha.max() + 1e-8)
    return alpha


# layers discretization for rules tuning
def discretize_percentile(alpha): #method 1 - percentile
    p25 = np.percentile(alpha, 25)
    p75 = np.percentile(alpha, 75)
    multipliers = np.ones_like(alpha)
    multipliers[alpha > p75] = 3
    multipliers[(alpha > p25) & (alpha <= p75)] = 2
    multipliers[alpha <= p25] = 1
    return multipliers.astype(int)

def discretize_proportional(alpha): #method 2 - proporzionale
    multipliers = 1 + np.round(alpha * 3)
    return int(multipliers)



def quantize_matrix(w):
    # matrix quantization for last layer of SNPS: Transform from real values to {-1,0,1}
    if Config.QUANTIZE_METHOD == 1: q = quantize_percentile(w, Config.M_SPARSITY, Config.M_POSITIVE) # Percentile-based
    elif Config.QUANTIZE_METHOD == 2: q = quantize_threshold(w, Config.M_THRESHOLD) # Threshold-based
    else: _, q, _ = quantize_twn(w) #Config.QUANTIZE_METHOD == 3
    return q

#https://www.emergentmind.com/topics/ternary-weight-networks-twns
#il link contiene info sulle reti ternarie TWN con pesi {-1,0,1}
def quantize_percentile(w, p_zero, p_pos):
    """
    Quantizza W in {-1,0,1} usando percentuali fisse.
    Layer-wise (colonna per colonna).
    p_neg + p_zero + p_pos = 1
    """
    p_neg = 1 - p_zero - p_pos
    w_q = np.zeros_like(w)
    n_rows, n_cols = w.shape

    for c in range(n_cols):
        col = w[:, c]
        sorted_idx = np.argsort(col)  # indice dei pesi ordinati
        n_neg = int(p_neg * n_rows)
        n_zero = int(p_zero * n_rows)
        # assegno -1 ai più piccoli
        w_q[sorted_idx[:n_neg], c] = -1
        # assegno 0 ai successivi
        w_q[sorted_idx[n_neg:n_neg+n_zero], c] = 0
        # assegno +1 ai restanti
        w_q[sorted_idx[n_neg+n_zero:], c] = 1

    return w_q

def quantize_threshold(w, k=0.5):
    """
    Quantizza W in {-1,0,1} usando threshold layer-wise.
    k: fattore moltiplicativo della media dei valori assoluti della colonna
    """
    w_q = np.zeros_like(w)
    n_rows, n_cols = w.shape

    for c in range(n_cols):
        col = w[:, c]
        t = k * np.mean(np.abs(col))
        w_q[:, c] = np.where(col > t, 1, np.where(col < -t, -1, 0))

    return w_q


def quantize_twn(w):
    """
    TWN-style ternarization.
    Minimizza ||W - αB||^2 con B ∈ {-1,0,1}.

    Applica il metodo per colonna.

    Returns:
        Q : matrice ternaria scalata (α * B)
        B : matrice pura {-1,0,1}
        alphas : scaling factors per colonna
    """

    Q = np.zeros_like(w, dtype=np.float64)
    B = np.zeros_like(w, dtype=np.int8)
    alphas = []

    for j in range(w.shape[1]):
        wj = w[:, j]

        # δ = 0.7 * E(|w|)
        delta = 0.7 * np.mean(np.abs(wj))

        # ternary mask
        b = np.zeros_like(wj)
        b[wj > delta] = 1
        b[wj < -delta] = -1

        # calcolo α solo sui pesi non zero
        if np.sum(np.abs(b)) > 0:
            alpha = np.mean(np.abs(wj[np.abs(wj) > delta]))
        else:
            alpha = 0.0

        q = alpha * b

        Q[:, j] = q
        B[:, j] = b
        alphas.append(alpha)

    return Q, B, np.array(alphas)



"""500 data with 1 epoch:
    realValue accuracy: 0.7280
    SNPS accuracy: 0.666
    SVM accuracy: 0.888
    -----------
    
    500 with 6 epochs:
    
    realValue accuracy: 0.8420
    SNPS accuracy: 0.832
    SVM accuracy: 0.888
    ------------
    
    2000 with 1 epochs:
    
    realValue accuracy: 0.8320
    SNPS accuracy: 0.8215
    SVM accuracy: 0.921
    ------------
    
    500 with 4 epochs and suffled data:
    realValue accuracy: 0.7020
    SNPS accuracy: 0.71
    SVM accuracy: 0.888
    --------
    
    3000 with svm on SNPS
    
    SNPS svm accuracy: 0.8803333333333333
    Perceptron accuracy: 0.8323
    SVM accuracy: 0.9266666666666666
    --------------
    

    3000 con QRANGE 16
    SNPS svm accuracy: 0.8793333333333333
    MULTI SNPS svm accuracy: 0.893
    MULTI SNPS logreg accuracy: 0.8843333333333333
    Perceptron accuracy: 0.8653
    SVM accuracy: 0.9306666666666666
    LOGREG accuracy: 0.9206666666666666
    --------------
    
    5000 con QRANGE 20 e svm on SNPS
    SNPS svm accuracy: 0.8842
    MULTI SNPS svm accuracy: 0.9004
    Perceptron accuracy: 0.8638
    SVM accuracy: 0.9414
    ----------------
    
    2000 Qrange 16
    SNPS svm accuracy: 0.879
    SNPS_IMP svm accuracy: 0.889
    svm accuracy: 0.9215
    SNPS logreg accuracy: 0.879
    SNPS_IMP logreg accuracy: 0.8945
    logreg accuracy: 0.921
    
    
TEST ON DIFFERENT IMPORTANCE/ALPHA - all with 2000 Qrange 16
------------------
#method 1 - percentile

SNPS svm accuracy: 0.879
SNPS_IMP svm accuracy: 0.889
svm accuracy: 0.9215
SNPS logreg accuracy: 0.879
SNPS_IMP logreg accuracy: 0.8945
logreg accuracy: 0.921

--------------
#method 2 - proporzionale con (alpha * 3)

------SNPS svm accuracy: 0.879
------SNPS_IMP svm accuracy: 0.886
------svm accuracy: 0.9215
------SNPS logreg accuracy: 0.879
------SNPS_IMP logreg accuracy: 0.8905
------logreg accuracy: 0.921


--------------
#method 2 - proporzionale con (alpha * 6)

------SNPS svm accuracy: 0.879
------SNPS_IMP svm accuracy: 0.8875
------svm accuracy: 0.9215
------SNPS logreg accuracy: 0.879
------SNPS_IMP logreg accuracy: 0.882
------logreg accuracy: 0.921


------------------------------------
#!! alpha 2 - separabilità - method 1 - IL MIGLIORE!

------SNPS_IMP svm accuracy: 0.892
------svm accuracy: 0.9215
------SNPS_IMP logreg accuracy: 0.8945
------logreg accuracy: 0.921


----------------------------
#alpha 2 - separabilità - method 2 con (alpha * 3)

------SNPS_IMP svm accuracy: 0.8825
------svm accuracy: 0.9215
------SNPS_IMP logreg accuracy: 0.891
------logreg accuracy: 0.921


#--------------------#
#--------------------#
confronto il migliore (alpha 2 method 1) con più dati rispetto a alpha 1 e method 1

---alpha 1 - method 1; 5000 con QRANGE 30, QUANTIZE_METHOD = 1:

------SNPS_IMP svm accuracy: 0.8976
------ SNPS_IMP logreg ROC AUC: 0.9842738716226563
------svm accuracy: 0.9432
------SNPS_IMP logreg accuracy: 0.8752
------ SNPS_IMP logreg ROC AUC: 0.9822524209355377
------logreg accuracy: 0.9348

--------------------------

---alpha 2 - method 1; 5000 con QRANGE 30, QUANTIZE_METHOD = 1:
------SNPS_IMP svm accuracy: 0.9054
------svm accuracy: 0.9432
------SNPS_IMP logreg accuracy: 0.8732
------logreg accuracy: 0.9348

------------------------------

QUINDI ora mantengo alpha 2 method 1  -> abbasso QRANGE di molto
5000 dati QRANGE 5, QUANTIZE_METHOD = 1: IMPORTANTE: SVM è PEGGIORATA E LOGREG è MIGLIORATA
------SNPS_IMP svm accuracy: 0.8868
------ SNPS_IMP logreg ROC AUC: 0.9818811904418677
------svm accuracy: 0.9356
------SNPS_IMP logreg accuracy: 0.9086
------ SNPS_IMP logreg ROC AUC: 0.9834633919502526
------logreg accuracy: 0.9404

-------------------------
5000 dati QRANGE 10, QUANTIZE_METHOD = 1:

------SNPS_IMP svm accuracy: 0.8942
------ SNPS_IMP logreg ROC AUC: 0.9835764677706512
------svm accuracy: 0.942
------SNPS_IMP logreg accuracy: 0.8918
------ SNPS_IMP logreg ROC AUC: 0.9836198235252503
------logreg accuracy: 0.9384


#-----------------------#
#-----------------------#
ho aggiunto che test != train, valori nel config
#-----------------------#
#-----------------------#
Confronto QUANTIZE_METHOD 2 e 3, per ora avevo solo 1.


5000 dati 1000 test QRANGE 5 QUANTIZE_METHOD 2:

------SNPS_IMP svm accuracy: 0.911
ROC AUC: 0.9888781786887512
------svm accuracy: 0.944
------SNPS_IMP logreg accuracy: 0.926
ROC AUC: 0.986478484946012
------logreg accuracy: 0.958

!uguale a sopra ma 5000 nel test (come nei casi precedenti):


!!!ATTENZIONE: FINO A QUI HO SBAGLIATO E AVEVO svm", svm.coef_, get_importance(logreg.coef_))!
--------------------
5000 dati 1000 test QRANGE 30:



--------------------------

    """
