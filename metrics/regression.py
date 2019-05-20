import numpy as np

from utils import check_dims


def mean_squared_error(y: np.array, y_pred: np.array) -> np.float:
    y, y_pred = check_dims(y, y_pred)
    return np.mean(np.square((y_pred - y)))


def mean_absolute_error(y: np.array, y_pred: np.array) -> np.float:
    y, y_pred = check_dims(y, y_pred)
    return np.mean(np.abs((y_pred - y)))


def root_mean_squared_error(y: np.array, y_pred: np.array) -> np.float:
    y, y_pred = check_dims(y, y_pred)
    return np.sqrt(np.mean(np.square((y_pred - y))))


def r2(y: np.array, y_pred: np.array) -> np.float:
    y, y_pred = check_dims(y, y_pred)
    return 1 - mean_squared_error(y, y_pred) / y.std()


def quantile_loss(y: np.array, y_pred: np.array, t: np.float = 0.5) -> np.float:
    y, y_pred = check_dims(y, y_pred)
    n = y.shape[0]
    loss = np.sum(list(map(lambda y, y_pred: ((t - 1) * y < y_pred + t * y >= y_pred) * (y - y_pred), y, y_pred))) / n
    return loss
