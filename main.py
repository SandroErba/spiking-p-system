'''How to read the output table:
r: rule applied. r:999a+2;3->0;1 mean "using 3 charge, 0 spike are fired, with 1 delay". ! = firing rules.
1a is the condition part that follows: if c %999 == 2, the rule applies
c: internal charge of neuron. i:x(y) means "x charge received from neuron y"

Remember: a neuron has rules of type E/r^x->a. E is a regular expression that should find an EXACT match with the number of spikes a in the neuron.
the rule consumes x spikes, or all the spikes if x is no defined. We use div and mod to describe E, and source is x.
For the rules that want exact numbers, not regulars expressions, we are using div = k as "large number"
'''
from sps import MedMnist

#MedMnist.blood_SNPsystem_csv() # create the SNPS with 841 neurons for medmnist classification
#MedMnist.compute_blood_mnist(100, "compute")
MedMnist.launch_blood(1000, 0.5)

#TODO salvare il modello dopo un pruning lungo, provare G e B e unire
# CAMBIARE TRAIN E TEST SET

#TODO sistemare stampe, tunare crescita e decrescita, distruggere sinapsi e testare
# chiedere a chatty di sistemare codice, minimizzando l'uso delle stringhe, ma senza ripetere codice
# rende le matrici e in generale l'algoritmo scalabile, non fisso a 49 neuroni e 8 classi
# per ribilanciare potrei, per ogni classe, tenere solo il 50% dei pesi più alti indipendentemente dal loro valore,
# per astrarre dal numero di istanze di quella classe

# - step 0: 1° immagine spara, layer 1 riceve 1°
# - step 1: layer 1 legge 2° immagine e spara la 1°, layer 2 riceve spike 1° immagine
# - step 2: layer 1 legge 3° immagine e spara la 2°, layer 2 riceve la 2° e spara la 1°, layer 3 riceve la 1°
# - step 3: layer 1 legge 4° immagine e spara la 3°, layer 2 riceve la 3° e spara la 2°, layer 3 riceve la 2° e spara output della 1°

# IDEE: -potrei fare un segnale che parte dal centro e si diffonde finchè trova pixel, per avere un idea della grandezza
    # -oppure un metodo per capire il contorno dell'immagine, tipo segmentazione già fatta
    # -devo calcolare che ho già il primo strato di input, posso usarlo a mio piacere facendogli sparare verso più neuroni
    # -potrei anche collegare a un unico neurone addetto al conteggio dei pixel, che poi spara agli output