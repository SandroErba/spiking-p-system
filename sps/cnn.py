#PRIMA: search for an easy CNN and try to recreate it -> https://pythonguides.com/simple-mnist-convnet-keras/
# usare e mantenere aggiornati le test classes - lanciarli in pipeline con le push?
#ORA:
#LEGGERE DN-SNP: SNP-based lightweight deep network for CT image segmentation of COVID-19 o paper simili con CNN
#POI continuare con la creazione e modifica della rete sotto capendo come fare il (o i) kernel

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

import csv
from sps.config import Config
from sps.handle_image import get_mnist_data


def launch_CNN_SNPS():
    (train_red, train_green, train_blue, train_labels), \
    (test_red, test_green, test_blue, test_labels) = get_mnist_data('retinamnist')

    # Group color channels
    train_channels = [train_red, train_green, train_blue]
    test_channels = [test_red, test_green, test_blue]

    predictions = []

    for train_data, test_data in zip(train_channels, test_channels):

        cnn_SNPS_csv()                    # prepare CSV for this channel
        #rules_train_SNPS(train_data)            # adapt firing rules (layer 2)
        #syn_train_SNPS(train_data, train_labels)  # prune + inhibit
        #pred = compute_SNPS(test_data)          # test P system
        #predictions.append(pred)

    # Unpack predictions
    red_pred, green_pred, blue_pred = predictions

    #combined_ranking_score(red_pred, green_pred, blue_pred, test_labels)

def cnn_SNPS_csv(filename="csv/" + Config.CSV_NAME):
    """Generate the SN P system to replicate the cnn"""
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # Layer 1: Input RGB from 28x28 to 14x14 using 2x2 blocks
        for neuron_id in range(Config.NEURONS_LAYER1):
            block_row = (neuron_id // Config.IMG_SHAPE) // Config.BLOCK_SHAPE
            block_col = (neuron_id % Config.IMG_SHAPE) // Config.BLOCK_SHAPE
            block_id = block_row * int(Config.IMG_SHAPE/Config.BLOCK_SHAPE) + block_col
            output_neuron = Config.NEURONS_LAYER1 + block_id

            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                f"[{output_neuron}]", # output_targets
                0,                    # neuron_type
                "[0,4,4,4,0]",
                "[0,3,3,3,0]",
                "[0,2,2,2,0]",
                "[0,1,1,1,0]"         # firing rules
            ])

        # Layer 2: Pooling (49 neurons)
        output_targets = str(list(range(Config.NEURONS_LAYER1_2, Config.NEURONS_TOTAL))) # for firing at the output neurons
        for neuron_id in range(Config.NEURONS_LAYER1, Config.NEURONS_LAYER1_2):
            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                output_targets,       # output_targets
                1,                    # neuron_type
                "[1,13,1,4,0]",
                "[1,9,1,3,0]",
                "[1,5,1,2,0]",
                "[1,1,1,1,0]",       # firing rules if c >= 1
                "[1,1,1,0,0]"        # forgetting rule if didn't fire
            ])

        # Layer 3: Output (8 neurons)
        for neuron_id in range(Config.NEURONS_LAYER1_2, Config.NEURONS_TOTAL):
            #label = neuron_id - Config.NEURONS_LAYER1_2
            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                "[]",                 # output_targets
                2,                    # neuron_type
                "[1,1,1,0,0]"         # forgetting rule
            ])