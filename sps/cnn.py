#PRIMA: search for an easy CNN and try to recreate it -> https://pythonguides.com/simple-mnist-convnet-keras/
#LEGGERE DN-SNP: SNP-based lightweight deep network for CT image segmentation of COVID-19 o paper simili con CNN
#LEGGERE PER STDP: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full

# for cnn and kernels see https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
#guardare il lavoro di Iris Ermini, analizza immagini grandi binarizzandole

#posso provare con i GA (o simili) per trainare i kernel?
#dopo ogni iterazione posso mostrare i risultati dei filtri su immagini con pattern chiari, per vedere cosa stanno imparando
#magari non raggiungo buone performance, ma studio il funzionamento e le diverse applicazioni di GA o simili
#posso tenere traccia dei valori di exploration e exploitation calcolando le performance ottenute e la differenza nelle regole presenti
#tunare regole e neuroni assieme? uno alla volta?
#visto che sono non deterministici, come ACO e PSO, posso dire che le immagini sono incapsulate e non leggibili perchè non puoi fare backtraking

import numpy as np
from matplotlib import pyplot as plt

from sps.digit_image import get_digit_data, get_28_digit_data
from sps.handle_csv import cnn_SNPS_csv
from sps.config import Config
from sps.med_image import get_mnist_data
from sps.snp_system import SNPSystem

#dataset = 'medmnist' #can be digit, medmnist, flower

def launch_28_CNN_SNPS():
    x_train, _, _, _ = get_28_digit_data()
    cnn_SNPS_csv() #use only if the csv was changed
    compute_cnn(x_train)


def compute_cnn(train_data):
    snps = SNPSystem(5, Config.TRAIN_SIZE + 5, "images", "images", True)
    snps.load_neurons_from_csv("csv/" + Config.CSV_NAME)
    snps.spike_train = train_data
    snps.start()

    #print(snps.feature_image[0])
    #print("image 0: ", snps.feature_image[0])
    #print("image shape: ", snps.feature_image.shape)

    #show_results(train_data, snps.feature_image, Config.SHAPE_FEATURE, Config.K_RANGE) #show output images for feature extraction layer
    #avg_pooling_image = snps.pooling_image // 4 #TODO this is the work that the firing rules in layer pool will do
    #TODO the range is still the same because i sum 4 values, and the do an avg pooling, so /4
    #show_results(train_data, avg_pooling_image, Config.SHAPE_POOL, Config.K_RANGE) #show output images for average pooling layer


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

