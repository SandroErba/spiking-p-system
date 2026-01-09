#PRIMA: search for an easy CNN and try to recreate it -> https://pythonguides.com/simple-mnist-convnet-keras/
# usare e mantenere aggiornati le test classes - lanciarli in pipeline con le push?
#LEGGERE DN-SNP: SNP-based lightweight deep network for CT image segmentation of COVID-19 o paper simili con CNN
#LEGGERE PER STDP: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full
#TODO pushare su master e aggiornare Fra con handwritten
#POI continuare con la creazione e modifica della rete capendo come tunare i kernel

#start from the 28x28 -> 26x26xN with N kernels.
#i have to create the kernels, copy from existing ones, and, if i can, train them.
#print the resulting 26x26 images and see if some pattern are appearing
#increase the number of kernels, improve them to increase the learned patterns
# see https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148

#GA - ?? posso provare con i GA (o simili) per trainare i kernel??
#dopo ogni iterazione posso mostrare i risultati dei filtri su immagini con pattern chiari, per vedere cosa stanno imparando
#magari non raggiungo buone performance, ma studio il funzionamento e le diverse applicazioni di GA o simili
#posso tenere traccia dei valori di exploration e exploitation calcolando le performance ottenute e la differenza nelle regole presenti
#tunare regole e neuroni assieme? uno alla volta?
#evolvere i parametri durante le run
#visto che sono non deterministici, come ACO e PSO, posso dire che le immagini sono incapsulate e non leggibili perchè non puoi fare backtraking

#guardare il lavoro di Iris Ermini, analizza immagini grandi binarizzandole
import numpy as np
from matplotlib import pyplot as plt

from sps.handle_csv import cnn_SNPS_csv
from sps.config import Config
from sps.med_image import get_mnist_data
from sps.snp_system import SNPSystem


def launch_CNN_SNPS():
    (train_red, train_green, train_blue, train_labels), (_) = get_mnist_data('bloodmnist')

    train_channels = [train_red, train_green, train_blue] # Group color channels
    #test_channels = [test_red, test_green, test_blue]
    #predictions = []

    cnn_SNPS_csv() #use only if the csv was changed
    for train_data in zip(train_channels):
        compute_cnn(train_data[0])
        #rules_train_SNPS(train_data)              # adapt firing rules (layer 2)
        #syn_train_SNPS(train_data, train_labels)  # prune + inhibit
        #pred = compute_SNPS(test_data)            # test P system
        #predictions.append(pred)


    #combined_ranking_score(red_pred, green_pred, blue_pred, test_labels)

def compute_cnn(train_data):
    snps = SNPSystem(5, Config.TRAIN_SIZE + 5, "images", "images", True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)
    snps.spike_train = train_data
    snps.start()

    #print(snps.image_output[0]) #TODO delete

    show_feature(snps.image_output)


def show_feature(output_array):
    images = np.asarray(output_array)
    num_inputs = images.shape[1]
    kernels = Config.KERNEL_NUMBER

    cols = min(kernels, 5)               # kernels per row
    rows = int(np.ceil(kernels / cols))  # rows needed

    plt.figure(figsize=(3 * cols, 3 * rows * num_inputs))

    img_index = 0

    for inp in range(num_inputs):
        for k in range(kernels):
            start = k * (Config.SEGMENTED_SHAPE**2)
            end   = (k+1) * (Config.SEGMENTED_SHAPE**2)

            feature = images[start:end, inp].reshape(
                (Config.SEGMENTED_SHAPE, Config.SEGMENTED_SHAPE)
            )

            # automatic scaling based on actual values in the feature map
            vmin = np.min(feature)
            vmax = np.max(feature)

            plt.subplot(num_inputs * rows, cols, img_index + 1)
            plt.imshow(feature, cmap="gray", vmin=vmin, vmax=vmax)
            plt.title(f"Input {inp} – Kernel {k}")
            plt.axis("off")

            img_index += 1

    plt.tight_layout()
    plt.show()
