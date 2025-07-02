'''How to read the output table:
r: rule applied. r:999a+2;3->0;1 mean "using 3 charge, 0 spike are fired, with 1 delay". ! = firing rules.
1a is the condition part that follows: if c %999 == 2, the rule applies
c: internal charge of neuron. i:x(y) means "x charge received from neuron y"

Remember: a neuron has rules of type E/r^x->a. E is a regular expression that should find an EXACT match with the number of spikes a in the neuron.
the rule consumes x spikes, or all the spikes if x is no defined. We use div and mod to describe E, and source is x.
For the rules that want exact numbers, not regulars expressions, we are using div = k as "large number"
'''
from sps import MedMnist

#MedMnist.create_blood_network_csv() # create the SNPS with 841 neurons for medmnist classification
MedMnist.rules_train_rgb_blood_mnist()
#Networks.compute_divisible_3()

# - step 0: 1° immagine spara, layer 1 riceve 1°
# - step 1: layer 1 legge 2° immagine e spara la 1°, layer 2 riceve spike 1° immagine
# - step 2: layer 1 legge 3° immagine e spara la 2°, layer 2 riceve la 2° e spara la 1°, layer 3 riceve la 1°
# - step 3: layer 1 legge 4° immagine e spara la 3°, layer 2 riceve la 3° e spara la 2°, layer 3 riceve la 2° e spara output della 1°

# TODO algoritmo doppio: si alzano/abbassano pesi di chi spara per ogni classificazione,
    # TODO e si alzano/abbassano rules per neuroni che sparano tanto/poco.
    # prima sistemo i valori di firing delle regole, così non faccio un training inutile sulle sinapsi

# devo: -dividere i tre canali altrimenti mescolo i training -costruire la matrice dei pesi 0.5 e aggiornarla.
# potrei: nei bordi mettere un numero basso di spike richiesti, tipo 2 o 3, mentre al centro devono essere alti per sparare.
# -> per ora facciamo che spari sempre se hai 5 o più. poi le metterò dipendenti dalla zona
# sono fully connected, ma le classificazioni sbagliate distruggono le sinapsi, mentre quelle corette le "rinforzano"
# potrei fare che ogni sinapse ha uno spessore, che ne descrive la forza. nella fase di train varia da 0 a 1, e alla fine
# resta o viene distrutto se questo è < 0 > di 0.5 (dovrebbe essere plasticità on sinaptica)
# è importante la posizionalità, non basta una fully connected, altrimenti conto solo il numero di pixel attivi senza forma
# inizio con 784 → 49 → 8 dove è tutto FC; lancio un immagine e vedo cosa classifica, aggiorno pesi di conseguenza.
# attenzione: capire BENE quali pesi aumentare e quali diminuire. ciclo sulle immagini, e alla fine eliminerò molte sinapsi
# ora parto dal canale Red con già 8 output.