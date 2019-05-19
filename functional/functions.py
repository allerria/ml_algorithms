from typing import Union

import numpy as np


def sigmoid(x: Union[np.array, np.float]) -> Union[np.array, np.float]:
    return 1 / (1 + np.exp(-x))


def softmax(x: np.array) -> np.array:
    x -= np.max(np.max(x))
    probs = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return probs
