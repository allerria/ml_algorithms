import numpy as np

from utils import check_dims


def accuracy(y: np.array, y_pred: np.array) -> np.float:
    y, y_pred = check_dims(y, y_pred)
    return np.sum(y == y_pred) / y.shape[0]


def precision(y: np.array, y_pred: np.array) -> np.float:
    y, y_pred = check_dims(y, y_pred)
    tp = np.sum(y == 1 & y_pred == 1)
    fp = np.sum(y != 1 & y_pred == 1)
    return tp / (tp + fp)


def recall(y: np.array, y_pred: np.array) -> np.float:
    y, y_pred = check_dims(y, y_pred)
    tp = np.sum(y == 1 & y_pred == 1)
    fn = np.sum(y == 1 & y_pred != 1)
    return tp / (tp + fn)


def fbeta(y: np.array, y_pred: np.array, beta: np.float) -> np.float:
    y, y_pred = check_dims(y, y_pred)
    p = precision(y, y_pred)
    r = recall(y, y_pred)
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def f1(y: np.array, y_pred: np.array) -> np.float:
    return fbeta(y, y_pred, 1)
