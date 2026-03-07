import numpy as np
from sps.config import Config


class OnlineDiscretePerceptron:
    def __init__(self, n_features=1352, n_classes=10, sparsity=0.5):
        # sparsity   : percentage of weights that must be 0
        np.random.seed(42)
        self.n_features = Config.NEURONS_LP #TODo those are not working, i cant hardcode the values in the declaration
        self.n_classes = Config.CLASSES
        self.sparsity = sparsity

        # random initialization with -1,0, or 1
        self.W = np.random.choice([-1, 0, 1], size=(n_features, n_classes))

    # ----------------------------
    # Forward step singolo
    # ----------------------------
    def forward(self, x):
        """
        Calcola i punteggi per 10 classi
        x: array shape (1352,)
        """
        return x @ self.W

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

        scores = self.forward(x)
        pred = np.argmax(scores) #TODO replace this 2 lines with the "predict" function

        if pred != y_true:
            # aggiornamento locale stile perceptron
            self.W[:, y_true] += x
            self.W[:, pred] -= x

        # proietta i pesi su {-1,0,1} e applica sparsity
        self._project_weights()

    # ----------------------------
    # Proiezione pesi discreti
    # ----------------------------
    def _project_weights(self):
        # clamp a [-1,1] TODO is correct? im no losing a lot of information, if numbers are >> 1?
        self.W = np.clip(self.W, -1, 1)

        # applica sparsity (più vicini allo 0 diventano 0)
        if self.sparsity > 0:
            flat = self.W.flatten()
            n_zero = int(self.sparsity * flat.size)

            # trova i pesi più piccoli in valore assoluto
            idx = np.argsort(np.abs(flat))[:n_zero]
            flat[idx] = 0
            self.W = flat.reshape(self.W.shape)

    # ----------------------------
    # Metodo finale: matrice sinapsi
    # ----------------------------
    def get_synapses(self):
        """
        Restituisce la matrice dei pesi discreti finale
        """
        return self.W.copy()


"""
In main:
# inizializzo modello
model = OnlineDiscretePerceptron(sparsity=0.6)
# simuliamo flusso dati
for t in range(3):  # 1000 time steps
    x_t = np.random.randint(0, 49, size=(1352,))
    print(x_t.shape)
    y_t = np.random.randint(0, 10)

    # aggiorno il perceptrone con il nuovo vettore
    model.update(x_t, y_t) #TODO find the place in the code where call this function

    # opzionale: puoi predire
    pred_class, scores = model.predict(x_t)

# alla fine
W_finale = model.get_synapses()
"""