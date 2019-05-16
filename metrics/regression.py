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
