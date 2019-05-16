# гоша гей
import numpy as np
from metrics import mean_squared_error


class LinearRegressor:
    def __init__(self) -> None:
        self.W = None


    def fit(self, X: np.array, y: np.array, method: str='sgd', lr: np.float=0.02,
    max_epochs: np.int=10000, eps: np.float=1e-8) -> None:
        self.W = np.zeros(X.shape[1])
        if method == 'analytic':
            self.W = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
            return None
        for e in range(max_epochs):
            prev_W = self.W.copy()
            idx = np.random.choice(X.shape[0], 1)[0]
            self.W -= 2 * lr * X[idx].T.dot(X[idx].dot(self.W) - y[idx])
            if np.linalg.norm(self.W - prev_W) < eps:
                break
        return None


    def predict(self, X: np.array) -> np.array:
        y_pred = X.dot(self.W)
        return y_pred


    def score(self, X: np.array, y: np.array) -> np.float:
        y_pred = self.predict(X)
        score = mean_squared_error(y, y_pred)
        return score
