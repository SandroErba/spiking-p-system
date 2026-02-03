#PRIMA: search for an easy CNN and try to recreate it -> https://pythonguides.com/simple-mnist-convnet-keras/
# usare e mantenere aggiornati le test classes - lanciarli in pipeline con le push?
#LEGGERE DN-SNP: SNP-based lightweight deep network for CT image segmentation of COVID-19 o paper simili con CNN
#LEGGERE PER STDP: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full
#POI continuare con la creazione e modifica della rete capendo come tunare i kernel

#start from the 28x28 -> 26x26xN with N kernels.
#I have to create the kernels, copy from existing ones, and, if i can, train them.
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

from sps.digit_image import get_digit_data, get_28_digit_data
from sps.handle_csv import cnn_SNPS_csv
from sps.config import Config
from sps.med_image import get_mnist_data
from sps.snp_system import SNPSystem

#dataset = 'medmnist' #can be digit, medmnist, flower

def launch_28_CNN_SNPS():
    train_final, _, _, _ = get_28_digit_data()
    cnn_SNPS_csv() #use only if the csv was changed
    compute_cnn(train_final)


def compute_cnn(train_data):
    snps = SNPSystem(5, Config.TRAIN_SIZE + 5, "images", "images", True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)
    snps.spike_train = train_data
    snps.start()

    #print(snps.image_output[0])
    #print("image 0: ", snps.image_output[0])
    #print("image shape: ", snps.image_output.shape)
    show_feature(train_data, snps.image_output) #show output images


def show_feature(train_data, output_array):
    images = np.asarray(output_array)
    num_inputs = images.shape[1]
    kernels = Config.KERNEL_NUMBER

    cols = min(kernels + 1, 6)            # +1 for original image
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
            start = k * (Config.SEGMENTED_SHAPE**2)
            end   = (k + 1) * (Config.SEGMENTED_SHAPE**2)

            feature = images[start:end, inp].reshape(
                Config.SEGMENTED_SHAPE, Config.SEGMENTED_SHAPE
            )

            vmin = np.min(feature)
            vmax = np.max(feature)


            #TODO i can use min and max range from numbers of -1 and 1 in kernels, they resemble
            print("IMAGE: ", img_index , "KERNEL: " , k)
            print("---vmin: ", vmin , "---vmax: " , vmax)


            plt.subplot(num_inputs * rows, cols, img_index + 1)
            plt.imshow(feature, cmap="gray", vmin=vmin, vmax=vmax)
            plt.title(f"Input {inp} – Kernel {k}")
            plt.axis("off")

            img_index += 1

    plt.tight_layout()
    plt.show()

