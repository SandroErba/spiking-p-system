'''How to read the output table:
r: rule applied. r:999a+2;3->0;1 mean "using 3 charge, 0 spike are fired, with 1 delay". ! = firing rules.
1a is the condition part that follows: if c %999 == 2, the rule applies
c: internal charge of neuron. i:x(y) means "x charge received from neuron y"

Remember: a neuron has rules of type E/r^x->a. E is a regular expression that should find an EXACT match with the number of spikes a in the neuron.
the rule consumes x spikes, or all the spikes if x is no defined. We use div and mod to describe E, and source is x.
For the rules that want exact numbers, not regulars expressions, we are using div = k as "large number"
'''
from sps import Networks
import csv

def create_blood_network_csv(filename="neurons784image.csv"):
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "initial_charge", "output_targets", "neuron_type", "rules"])

        # Layer 1: Input RGB (784 neurons) from 28x28 to 7x7 using 4x4 blocks
        for neuron_id in range(784):
            row = neuron_id // 28
            col = neuron_id % 28
            block_row = row // 4
            block_col = col // 4
            block_id = block_row * 7 + block_col
            output_neuron = 784 + block_id

            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                f"[{output_neuron}]", # output_targets
                0,                    # neuron_type
                "[0,1,1,1,0]"         # firing rule
            ])

        # Layer 2: Pooling (49 neurons) - id 784–832
        for neuron_id in range(784, 784 + 49):
            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                "[833, 834, 835, 836, 837, 838, 839, 840]", # output_targets
                1,                    # neuron_type
                "[1,0,5,1,0]"         # firing rule if c >= 5 TODO così non le spara tutte e restano in memoria tra gli steps
            ])

        # Layer 3: Output (8 neurons) - id 833–840
        for neuron_id in range(833, 841):
            label = neuron_id - 833
            writer.writerow([
                neuron_id,            # id
                0,                    # initial_charge
                "[]", # output_targets
                2,                    # neuron_type
                "[1,0,0,0,0]"         # forgetting rule
            ])

    print(f"Created {filename} with 841 neurons")

#create_blood_network_csv() create the SNPS with 841 neurons for medmnist classification
Networks.compute_blood_mnist()


# TODO salvare il file history in modo che sia leggibile e permanente.
#  fare 3 reti parallele e identiche inizialmente, una per ogni canale colore
# -> per ora facciamo che spari sempre se hai 5 o più. poi le metterò dipendenti dalla zona
# sono fully connected, ma le classificazioni sbagliate distruggono le sinapsi, mentre quelle corette le "rinforzano"
# potrei fare che ogni sinapse ha uno spessore, che ne descrive la forza. nella fase di train varia da 0 a 1, e alla fine
# resta o viene distrutto se questo è < 0 > di 0.5 (dovrebbe essere plasticità on sinaptica)
# è importante la posizionalità, non basta una fully connected, altrimenti conto solo il numero di pixel attivi senza forma
# inizio con 784 → 49 → 8 dove è tutto FC; lancio un immagine e vedo cosa classifica, aggiorno pesi di conseguenza.
# attenzione: capire BENE quali pesi aumentare e quali diminuire. ciclo sulle immagini, e alla fine eliminerò molte sinapsi
# ora parto dal canale Red con già 8 output. su chatty ho il codice che genera il csv, lo adatto e lo proviamo.
# gli input delle immagini sono ancora da testare, potrebbero essere sbagliati