import numpy as np
from base import BaseModel, ClassifierMixin
from functional import softmax


#linear kernel
class SVM(BaseModel, ClassifierMixin):


    def __init__(self, C: np.float=1.0, lr: np.float=0.02,
                max_epochs: np.int=1000, eps: np.float=1e-8) -> None:
        self.W = None
        self.C = C
        self.lr = lr
        self.max_epochs = max_epochs
        self.eps = eps


    def fit(self, X: np.array, y: np.array) -> None:
        n_train, dim = X.shape
        n_classes = np.max(y) + 1
        self.W = np.zeros((dim, n_classes))

        for e in range(self.max_epochs):
            prev_W = self.W.copy()
            dW = np.zeros_like(self.W)

            scores = X.dot(self.W)
            probs = softmax(scores)

            dscores = probs
            dscores[range(n_train), y] -= 1

            dW = X.T.dot(dscores)
            dW /= n_train
            dW += 2 * self.C * self.W
            self.W -= self.lr * dW

            if np.linalg.norm(self.W - prev_W) < self.eps:
                break
