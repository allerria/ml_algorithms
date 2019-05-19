import numpy as np

from base import BaseModel, ClassifierMixin
from functional import sigmoid


class LogisticRegression(BaseModel, ClassifierMixin):

    def __init__(self, C: np.float = 1.0, lr: np.float = 0.02,
                 max_epochs: np.int = 1000, eps: np.float = 1e-8) -> None:
        self.W = None
        self.C = C
        self.lr = lr
        self.max_epochs = max_epochs
        self.eps = eps

    def fit(self, X: np.array, y: np.array) -> None:
        n_train = X.shape[0]
        self.W = np.zeros(X.shape[1])
        for e in range(self.max_epochs):
            prev_W = self.W.copy()
            scores = sigmoid(X.dot(self.W))
            dW = X.T.dot(scores - y) / n_train
            dW += 2 * self.C * self.W
            self.W -= self.lr * dW
            if np.linalg.norm(self.W - prev_W) < self.eps:
                break

    def raw_predict(self, X: np.array) -> np.array:
        scores = sigmoid(super().raw_predict(X))
        return scores
