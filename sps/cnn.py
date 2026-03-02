#PRIMA: search for an easy CNN and try to recreate it -> https://pythonguides.com/simple-mnist-convnet-keras/
#LEGGERE DN-SNP: SNP-based lightweight deep network for CT image segmentation of COVID-19 o paper simili con CNN
#LEGGERE PER STDP: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full

# for cnn and kernels see https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
#guardare il lavoro di Iris Ermini, analizza immagini grandi binarizzandole

import numpy as np
from sps.digit_image import get_28_digit_data
from sps.handle_csv import cnn_SNPS_csv, extend_csv
from sps.config import Config
from sps.snp_system import SNPSystem
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def launch_28_CNN_SNPS():
    x_train, y_train, x_test, y_test = get_28_digit_data()
    cnn_SNPS_csv() #use only if the csv was changed
    w, svm, rf = train_cnn(x_train, y_train)

    #q_perc = quantize_matrix(w)
    """
    test_cnn(x_test, y_test, q_perc, "q_perc")
    test_cnn(x_test, y_test, q_thres, "q_thres")
    test_cnn(x_test, y_test, q_twn, "q_twn")"""

    return test_cnn(x_test, y_test, w, "q_perc", svm)


def train_cnn(x_train, y_train):
    snps = SNPSystem(5, Config.TRAIN_SIZE + 5, True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)
    snps.spike_train = x_train
    snps.labels = y_train

    snps.start()
    #for epoch in range(4):
    #    snps.t_step = 0
    #    print("Epoch", epoch)
    #    snps.start()

    #SVM
    svm = LinearSVC(C=1.0, max_iter=10000)
    svm.fit(snps.pooling_image.T, y_train)

    #RANDOM FOREST
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(snps.pooling_image.T, y_train)

    #TODO NOW: try Logistic Regression

    return snps.model.get_synapses(), svm, rf


def test_cnn(x_test, y_test, w, q_name, svm):
    snps = SNPSystem(5, Config.TEST_SIZE + 5, True)
    snps.spike_train = x_test

    #TODO make this work together
    """print("testing the perceptron on SNPS")  
    q_perc = quantize_matrix(w)
    extended_path = extend_csv("csv/" + Config.CSV_NAME, np.array(q_perc), q_name)
    snps.load_neurons_from_csv(extended_path)
    snps.start()
    y_pred = np.argmax(snps.charge_map_prediction, axis=0)
    accuracy = np.mean(y_pred == y_test)
    print("SNPS perceptron accuracy:", accuracy)"""

    print("testing the svm on SNPS")
    q_svm = quantize_matrix(svm.coef_.T)
    extended_path = extend_csv("csv/" + Config.CSV_NAME, np.array(q_svm), "svm")
    snps.load_neurons_from_csv(extended_path)
    snps.start()
    y_pred = np.argmax(snps.charge_map_prediction, axis=0)
    accuracy = np.mean(y_pred == y_test)
    print("SNPS svm accuracy:", accuracy)


    features_int = snps.pooling_image // 4   # integer floor division
    features_int_pos = np.maximum(features_int, 0)
    acc_perc  = evaluate_accuracy(features_int_pos, y_test, w,  name="Perceptron")  #for compare the performance degradation

    #SVM
    acc_svm = svm.score(features_int_pos.T, y_test)
    print("SVM accuracy:", acc_svm)

    #TODO add Logistic Regression

    return accuracy



def quantize_matrix(w): #dovrò usare queste 3 matrici su 3 SNPS e controllarli sul test set
    #Transform from real values to {-1,0,1}
    q_perc = quantize_percentile(w, Config.SPARSITY, Config.POSITIVE) # Percentile-based
    return q_perc
    #q_thres = quantize_threshold(w, k=0.5) # Threshold-based #TODO use all 3 matrices and compare them
    #_, q_twn, _ = quantize_twn(w) # Threshold-based
    #analyze_quantized_matrix(q_perc, name="Percentile")
    #analyze_quantized_matrix(q_thres, name="Threshold")
    #analyze_quantized_matrix(q_twn, name="B_TWN")
    #Quantization error
    #print("Err Percentile:", np.linalg.norm(w - q_perc, ord='fro'))
    #print("Err Threshold:", np.linalg.norm(w - q_thres, ord='fro'))
    #print("Err TWN:", np.linalg.norm(w - q_twn, ord='fro'))
    #return q_perc, q_thres, q_twn



def evaluate_accuracy(features, labels, W, name="Model"):
    logits = W.T @ features
    preds = np.argmax(logits, axis=0)
    if len(preds) == len(labels):
        accuracy = np.mean(preds == labels)
        print(f"{name} accuracy: {accuracy:.4f}")
        return accuracy
    else:
        print("preds has size ", len(preds), "while labels has size ", len(labels))


#https://www.emergentmind.com/topics/ternary-weight-networks-twns
#il link contiene info sulle reti ternarie TWN con pesi {-1,0,1}
def quantize_percentile(W, p_zero, p_pos): #TODO ___trainable___ test some different p_neg and p_zero and check performance
    """
    Quantizza W in {-1,0,1} usando percentuali fisse.
    Layer-wise (colonna per colonna).
    p_neg + p_zero + p_pos = 1
    """
    p_neg = 1 - p_zero - p_pos
    W_q = np.zeros_like(W)
    n_rows, n_cols = W.shape

    for c in range(n_cols):
        col = W[:, c]
        sorted_idx = np.argsort(col)  # indice dei pesi ordinati
        n_neg = int(p_neg * n_rows)
        n_zero = int(p_zero * n_rows)
        # assegno -1 ai più piccoli
        W_q[sorted_idx[:n_neg], c] = -1
        # assegno 0 ai successivi
        W_q[sorted_idx[n_neg:n_neg+n_zero], c] = 0
        # assegno +1 ai restanti
        W_q[sorted_idx[n_neg+n_zero:], c] = 1

    return W_q

def quantize_threshold(W, k=0.5): #TODO ___trainable___ test different k and check performance
    """
    Quantizza W in {-1,0,1} usando threshold layer-wise.
    k: fattore moltiplicativo della media dei valori assoluti della colonna
    """
    W_q = np.zeros_like(W)
    n_rows, n_cols = W.shape

    for c in range(n_cols):
        col = W[:, c]
        t = k * np.mean(np.abs(col))
        W_q[:, c] = np.where(col > t, 1, np.where(col < -t, -1, 0))

    return W_q


def quantize_twn(W):
    """
    TWN-style ternarization.
    Minimizza ||W - αB||^2 con B ∈ {-1,0,1}.

    Applica il metodo per colonna.

    Returns:
        Q : matrice ternaria scalata (α * B)
        B : matrice pura {-1,0,1}
        alphas : scaling factors per colonna
    """

    Q = np.zeros_like(W, dtype=np.float64)
    B = np.zeros_like(W, dtype=np.int8)
    alphas = []

    for j in range(W.shape[1]):
        w = W[:, j]

        # δ = 0.7 * E(|w|)
        delta = 0.7 * np.mean(np.abs(w))

        # ternary mask
        b = np.zeros_like(w)
        b[w > delta] = 1
        b[w < -delta] = -1

        # calcolo α solo sui pesi non zero
        if np.sum(np.abs(b)) > 0:
            alpha = np.mean(np.abs(w[np.abs(w) > delta]))
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
    
    5000 con QRANGE 20 e svm on SNPS
    SNPS svm accuracy: 0.8842
    Perceptron accuracy: 0.8638
    SVM accuracy: 0.9412
    
    """
