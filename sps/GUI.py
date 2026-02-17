#vorrei un interfaccia grafica che permetta di utilizzare il simulatore senza mettere mano al codice
#con 3 modi per runnare:
# -su reti piccole tipo quelle di esempio (vedi other_networks.py) scegliendo quale si vuole far partire,
# -generando e runnando csv con immagini come input, vedi cnn.py o med_mnist.launch_quantized_SNPS() (il secondo è da sistemare).
# -definendo accuratamente i seguenti layer presenti nelle Convolutional Neural Networks: average pooling, kernelization, Fully connected.
    #in questo modo si può definire una sorta di CNN dalla GUI definendo quanti e quali layer si vogliono utilizzare.
    #(attualmente il codice deve ancora essere generalizzato per questa opzione)
    #per info sulle CNN vedi https://medium.com/data-science/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148

#per ora non renderei i parametri presenti in Config modificabili dall'interfaccia
#Francesca si occuperà della generazione dell'output, sarà solo da integrare e mostrare a schermo (e salvarlo come file)
