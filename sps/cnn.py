#PRIMA: search for an easy CNN and try to recreate it -> https://pythonguides.com/simple-mnist-convnet-keras/
#LEGGERE DN-SNP: SNP-based lightweight deep network for CT image segmentation of COVID-19 o paper simili con CNN
#LEGGERE PER STDP: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full

# for cnn and kernels see https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
#guardare il lavoro di Iris Ermini, analizza immagini grandi binarizzandole

#dopo ogni iterazione posso mostrare i risultati dei filtri su immagini con pattern chiari, per vedere cosa stanno imparando
#resta aperto il tuning delle regole, come si può effettuare?

import numpy as np
from matplotlib import pyplot as plt

from sps.digit_image import get_28_digit_data
from sps.handle_csv import cnn_SNPS_csv, extend_csv
from sps.config import Config
from sps.snp_system import SNPSystem

def launch_28_CNN_SNPS():
    x_train, y_train, x_test, y_test = get_28_digit_data()
    cnn_SNPS_csv() #use only if the csv was changed
    w = train_cnn(x_train, y_train)
    q_perc = quantize_matrix(w)


    #S_test_cnn(x_test, y_test, w)
    test_cnn(x_test, y_test, q_perc, "q_perc")

    """
    test_cnn(x_test, y_test, q_perc, "q_perc")
    test_cnn(x_test, y_test, q_thres, "q_thres")
    test_cnn(x_test, y_test, q_twn, "q_twn")"""


def train_cnn(x_train, y_train):
    snps = SNPSystem(5, Config.TRAIN_SIZE + 5, True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)
    snps.spike_train = x_train
    snps.labels = y_train
    snps.start()

    return snps.model.get_synapses()

    #show_results(x_train, snps.feature_image, Config.SHAPE_FEATURE, Config.K_RANGE) #show output images for feature extraction layer
    #avg_pooling_image = snps.pooling_image // 4
    #show_results(x_train, avg_pooling_image, Config.SHAPE_POOL, Config.K_RANGE) #show output images for average pooling layer


def test_cnn(x_test, y_test, q, q_name):
    extended_path = extend_csv("csv/" + Config.CSV_NAME, np.array(q), q_name)
    snps = SNPSystem(5, Config.TEST_SIZE + 5, True)
    snps.load_neurons_from_csv(extended_path)
    snps.spike_train = x_test
    snps.start()

    features = snps.pooling_image / 4.0
    print("range feature", features.shape)


    num_neg = 0
    for i in range(features.shape[0]):        # righe (neuroni)
        for j in range(features.shape[1]):    # colonne (immagini)
            if features[i][j] < 0:
                num_neg += 1
    print("Totale valori negativi :::", num_neg) #720 su 1000 train set
    #TODO NOW:: i negativi sono trascurabili, l'errore è da un'altra parte. confrontare features in input

    acc_perc  = evaluate_accuracy(features, y_test, q,  name="Percentile")

    y_pred = np.argmax(snps.charge_map_prediction, axis=0)
    accuracy = np.mean(y_pred == y_test)
    print("SNPS accuracy:", accuracy) #TODO check why its not the same of previous faster method


def quantize_matrix(w): #dovrò usare queste 3 matrici su 3 SNPS e controllarli sul test set
    #Transform from real values to {-1,0,1}
    q_perc = quantize_percentile(w, p_neg=0.4, p_zero=0.2, p_pos=0.4) # Percentile-based
    #q_thres = quantize_threshold(w, k=0.5) # Threshold-based
    #_, q_twn, _ = quantize_twn(w) # Threshold-based
    #print("Q_perc:", np.unique(q_perc, return_counts=True))
    #print("Q_thres:", np.unique(q_thres, return_counts=True))
    #print("B_twn:", np.unique(q_twn, return_counts=True))
    # Analisi matrici quantizzate
    #analyze_quantized_matrix(q_perc, name="Percentile")
    #analyze_quantized_matrix(q_thres, name="Threshold")
    #analyze_quantized_matrix(q_twn, name="B_TWN")

    #Quantization error TODO i can compare those matrices with some more detailed methods
    #print("Err Percentile:", np.linalg.norm(w - q_perc, ord='fro'))
    #print("Err Threshold:", np.linalg.norm(w - q_thres, ord='fro'))
    #print("Err TWN:", np.linalg.norm(w - q_twn, ord='fro'))
    #return q_perc, q_thres, q_twn
    return q_perc


def evaluate_accuracy(features, labels, W, name="Model"):
    """
    features: (n_features, n_samples)
    labels:   (n_samples,)
    W:        (n_features, n_classes)
    """
    logits = W.T @ features
    preds = np.argmax(logits, axis=0)
    accuracy = np.mean(preds == labels)
    print(f"{name} accuracy: {accuracy:.4f}")
    return accuracy

def show_results(train_data, output_array, img_size, img_range):
    print("range of values for current image:" , img_range)
    images = np.asarray(output_array)
    num_inputs = images.shape[1]
    kernels = Config.KERNEL_NUMBER

    #cols = min(kernels + 1, 6)            # +1 for original image
    cols = Config.KERNEL_NUMBER + 1
    rows = int(np.ceil((kernels + 1) / cols))

    plt.figure(figsize=(3 * cols, 3 * rows * num_inputs))

    img_index = 0
    for inp in range(num_inputs):

        plt.subplot(num_inputs * rows, cols, img_index + 1)
        #print("ORIGINAL IMAGE:", train_data[inp])
        plt.imshow(train_data[inp], cmap="gray")
        plt.title(f"Input {inp} – Original")
        plt.axis("off")

        img_index += 1

        for k in range(kernels):
            start = k * (img_size ** 2)
            end   = (k + 1) * (img_size ** 2)

            feature = images[start:end, inp].reshape(
                img_size, img_size
            )

            vmin, vmax = img_range[k]
            plt.subplot(num_inputs * rows, cols, img_index + 1)
            plt.imshow(feature, cmap="gray", vmin=vmin, vmax=vmax)
            plt.title(f"Input {inp} – Kernel {k}")
            plt.axis("off")

            #print("kernel number", k, "values",feature)

            img_index += 1

    plt.tight_layout()
    plt.show()


#https://www.emergentmind.com/topics/ternary-weight-networks-twns
#il link contiene info sulle reti ternarie TWN con pesi {-1,0,1}
def quantize_percentile(W, p_neg=0.4, p_zero=0.2, p_pos=0.4): #TODO test some different p_neg and p_zero and check performance
    """
    Quantizza W in {-1,0,1} usando percentuali fisse.
    Layer-wise (colonna per colonna).
    p_neg + p_zero + p_pos = 1
    """
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

def quantize_threshold(W, k=0.5): #TODO test 10 different k and check performance
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



"""Altri metodi possibili:

Discrete-State Training – Deng et al., 2017

Qui fanno qualcosa di radicale:

i pesi sono SEMPRE ternari

non esiste copia full precision

aggiornamenti tramite transizioni stocastiche tra stati {-1,0,1}

È più biologicamente plausibile."""



def analyze_quantized_matrix(Q, name="Q"):
    """
    Analizza una matrice quantizzata in {-1,0,1}.
    Q: numpy array shape (n_features, n_classes)
    name: string per etichettare la stampa
    """
    print(f"\n--- Analisi matrice {name} ---")
    n_rows, n_cols = Q.shape

    # Statistiche globali
    n_neg = np.sum(Q == -1)
    n_zero = np.sum(Q == 0)
    n_pos = np.sum(Q == 1)
    total = n_rows * n_cols

    print(f"Totale elementi: {total}")
    print(f"-1: {n_neg} ({100*n_neg/total:.2f}%)")
    print(f" 0: {n_zero} ({100*n_zero/total:.2f}%)")
    print(f"+1: {n_pos} ({100*n_pos/total:.2f}%)")
    print(f"Media globale: {np.mean(Q):.4f}, Media assoluta globale: {np.mean(np.abs(Q)):.4f}")

    # Statistiche per classe (colonna)
    for c in range(n_cols):
        col = Q[:, c]
        n_neg_c = np.sum(col == -1)
        n_zero_c = np.sum(col == 0)
        n_pos_c = np.sum(col == 1)
        mean_c = np.mean(col)
        mean_abs_c = np.mean(np.abs(col))
        print(f"Classe {c}: -1={n_neg_c} ({100*n_neg_c/n_rows:.2f}%), "
              f"0={n_zero_c} ({100*n_zero_c/n_rows:.2f}%), "
              f"+1={n_pos_c} ({100*n_pos_c/n_rows:.2f}%), "
              f"mean={mean_c:.2f}, mean_abs={mean_abs_c:.2f}")
