import numpy as np


class KMeans:
    def __init__(self, k: np.int, max_iter: np.int = 1000, eps: np.float = 1e-8) -> None:
        self.k = k
        self.centroids = None
        self.max_iter = max_iter
        self.eps = eps

    def predict(self, X: np.array) -> np.array:
        dists = np.zeros((X.shape[0], self.k))
        for i in range(X.shape[0]):
            for j in range(self.k):
                dists[i, j] = np.linalg.norm(X[i] - self.centroids[j])
        y = np.argmin(dists, axis=1)
        return y

    def metric(self, X: np.array):
        y = self.predict(X)
        score = np.sum([np.linalg.norm(self.centroids[y[i]] - X[i]) for i in range(y.shape[0])])
        return score

    def fit(self, X: np.array) -> None:
        self.centroids = X[np.random.choice(range(X.shape[0]), size=self.k, replace=False)]
        print(self.centroids)
        for e in range(self.max_iter):
            prev_score = self.metric(X)
            y = self.predict(X)
            for i in range(self.k):
                x_i = X[y == i]
                self.centroids[i] = np.mean(x_i, axis=0)
            score = self.metric(X)
            if np.abs(prev_score - score) < self.eps:
                break
