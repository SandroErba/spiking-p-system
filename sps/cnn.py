#PRIMA: search for an easy CNN and try to recreate it -> https://pythonguides.com/simple-mnist-convnet-keras/
# usare e mantenere aggiornati le test classes - lanciarli in pipeline con le push?
#ORA: sistemare config in modo che abbia 2 o 3 configurazione comode da cambiare
#POI continuare con la creazione e modifica della rete sotto capendo come fare il (o i) kernel
import csv
from sps.config import Config
from sps.handle_image import get_mnist_data
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



def launch_CNN_SNPS():
    (train_red, train_green, train_blue, train_labels), \
    (test_red, test_green, test_blue, test_labels) = get_mnist_data('retinamnist')

    # Group color channels
    train_channels = [train_red, train_green, train_blue]
    test_channels = [test_red, test_green, test_blue]

    predictions = []

    #for train_data, test_data in zip(train_channels, test_channels):

        #cnn_SNPS_csv()                    # prepare CSV for this channel
        #rules_train_SNPS(train_data)            # adapt firing rules (layer 2)
        #syn_train_SNPS(train_data, train_labels)  # prune + inhibit
        #pred = compute_SNPS(test_data)          # test P system
        #predictions.append(pred)

    # Unpack predictions
    #red_pred, green_pred, blue_pred = predictions

    #combined_ranking_score(red_pred, green_pred, blue_pred, test_labels)

