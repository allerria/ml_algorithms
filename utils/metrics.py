import numpy as np


def check_dims(y: np.array, y_pred: np.array) -> [np.array, np.array]:
    if y.ndim == 1:
        y = y.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    return y, y_pred
