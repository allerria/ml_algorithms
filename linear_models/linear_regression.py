import numpy as np
from base import BaseModel, RegressorMixin


class LinearRegressor(BaseModel, RegressorMixin):


    def __init__(self, lr: np.float=0.02, max_epochs: np.int=10000,
                 eps: np.float=1e-8) -> None:
        self.W = None
        self.lr = lr
        self.max_epochs = max_epochs
        self.eps = eps


    def fit(self, X: np.array, y: np.array, method: str='sgd') -> None:
        self.W = np.zeros(X.shape[1])
        if method == 'analytic':
            self.W = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
            return None
        for e in range(self.max_epochs):
            prev_W = self.W.copy()
            idx = np.random.choice(X.shape[0], 1)[0]
            self.W -= 2 * self.lr * X[idx].T.dot(X[idx].dot(self.W) - y[idx])
            if np.linalg.norm(self.W - prev_W) < self.eps:
                break
        return None
