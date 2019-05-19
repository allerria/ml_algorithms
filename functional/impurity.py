import numpy as np


def prob_classes(y: np.array) -> np.array:
    n_samples = y.shape[0]
    n_classes = np.max(y) + 1
    probs = np.array([np.sum(y == k) for k in range(n_classes)]) / n_samples
    return probs


def gini(y: np.array) -> np.float:
    probs = prob_classes(y)
    return np.sum(probs * (1 - probs))


def entropy(y: np.array) -> np.float:
    probs = prob_classes(y)
    probs += 1e-12
    return -np.sum(probs * np.log(probs))


def impurity(y: np.array, method: str = 'gini') -> np.float:
    if y.shape[0] == 0:
        return 0.0
    if method in {'gini', 'entropy'}:
        return globals()[method](y)
    return y.std()
