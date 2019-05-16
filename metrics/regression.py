import numpy as np


def mean_squared_error(y: np.array, y_pred: np.array) -> np.float:
    return np.sum((y_pred - y) ** 2)
