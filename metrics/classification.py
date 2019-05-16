import numpy as np
from utils import check_dims


def accuracy(y: np.array, y_pred: np.array) -> np.float:
    y, y_pred = check_dims(y, y_pred)
    return np.sum(y == y_pred) / y.shape[0]
