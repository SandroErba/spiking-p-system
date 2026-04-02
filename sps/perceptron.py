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
    # single forward pass
    # ----------------------------
    def forward(self, x):
        """
        
        x: array shape (1352,)
        """
        x_avg = x / 4.0
        return x_avg @ self.W

    def predict(self, x):
        """
        returns predicted class and raw scores
        """
        scores = self.forward(x)
        return np.argmax(scores), scores

    # ----------------------------
    # Update online
    # ----------------------------
    def update(self, x, y_true):
        """
        update weights based on a single example (x, y_true)
        x: array shape (1352,)
        y_true: correct label (0-9)
        """

        scores = self.forward(x)
        pred = np.argmax(scores)

        # ---- weight decay (L2) ----
        if self.weight_decay > 0:
            self.W *= (1 - self.weight_decay)

        if pred != y_true:
            self.W[:, y_true] += self.lr * x      # reinforce the correct class
            self.W[:, pred] -= self.lr * x        # weaken the incorrectly predicted class

    # ----------------------------
    # Final method: synapse matrix
    # ----------------------------
    def get_synapses(self):
        """
        Returns the final matrix of discrete weights after training.
        """
        return self.W.copy()
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
        update the weights of the perceptron step-by-step
        x: array shape (1352,)
        y_true: correct label (0-9)
        """

        scores = self.forward(x)
        pred = np.argmax(scores) #TODO replace this 2 lines with the "predict" function

        if pred != y_true:
            # local update like perceptron
            self.W[:, y_true] += x
            self.W[:, pred] -= x

        # project the weights onto {-1,0,1} and apply sparsity
        self._project_weights()

    # ----------------------------
    # Project weights
    # ----------------------------
    def _project_weights(self):
        # clamp a [-1,1] TODO is correct? im no losing a lot of information, if numbers are >> 1?
        self.W = np.clip(self.W, -1, 1)

        # apply sparsity (more similar to 0 become 0)
        if self.sparsity > 0:
            flat = self.W.flatten()
            n_zero = int(self.sparsity * flat.size)

            # find the smallest weights in absolute value
            idx = np.argsort(np.abs(flat))[:n_zero]
            flat[idx] = 0
            self.W = flat.reshape(self.W.shape)

    # ----------------------------
    # Final method: synapse matrix
    # ----------------------------
    def get_synapses(self):
        """
        Returns the final matrix of discrete weights after training.
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