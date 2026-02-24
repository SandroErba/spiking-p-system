import numpy as np
from sps.config import Config


class OnlineDiscretePerceptron:
    def __init__(self, n_features=1352, n_classes=10, sparsity=0.5):
        # sparsity   : percentage of weights that must be 0
        np.random.seed(35)
        self.n_features = Config.NEURONS_LP #TODo those variables are not working, i cant hardcode the values in the declaration
        self.n_classes = Config.CLASSES
        self.sparsity = sparsity

        # random initialization with -1,0, or 1
        self.W = np.random.choice([-1, 0, 1], size=(n_features, n_classes)).astype(np.float64)

    # ----------------------------
    # Forward step singolo
    # ----------------------------
    def forward(self, x):
        """
        Calcola i punteggi per 10 classi
        x: array shape (1352,)
        """
        x_avg = x / 4.0
        return x_avg @ self.W

    def predict(self, x):
        """
        Restituisce la classe predetta e i punteggi
        """
        scores = self.forward(x)
        return np.argmax(scores), scores

    # ----------------------------
    # Update online
    # ----------------------------
    def update(self, x, y_true):
        """
        Aggiorna i pesi del percettrone step-by-step
        x: array shape (1352,)
        y_true: label corretta (0-9)
        """
        #x_avg = x / 4.0
        scores = self.forward(x)
        pred = np.argmax(scores) #TODO replace this 2 lines with the "predict" function

        lr = 0.01
        if pred != y_true:
            self.W[:, y_true] += lr * x      # rinforza la classe corretta
            self.W[:, pred] -= lr * x        # indebolisce la classe predetta erroneamente

    # ----------------------------
    # Metodo finale: matrice sinapsi
    # ----------------------------
    def get_synapses(self):
        """
        Restituisce la matrice dei pesi discreti finale
        """
        return self.W.copy()