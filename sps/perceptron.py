import numpy as np

class OnlineDiscretePerceptron:
    def __init__(self, n_features, n_classes, lr,  weight_decay=0.0):
        # sparsity   : percentage of weights that must be 0
        np.random.seed(35)
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr

        self.weight_decay = weight_decay

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

        scores = self.forward(x)
        pred = np.argmax(scores)

        # ---- weight decay (L2) ----
        if self.weight_decay > 0:
            self.W *= (1 - self.weight_decay)

        if pred != y_true:
            self.W[:, y_true] += self.lr * x      # rinforza la classe corretta
            self.W[:, pred] -= self.lr * x        # indebolisce la classe predetta erroneamente

    # ----------------------------
    # Metodo finale: matrice sinapsi
    # ----------------------------
    def get_synapses(self):
        """
        Restituisce la matrice dei pesi discreti finale
        """
        return self.W.copy()