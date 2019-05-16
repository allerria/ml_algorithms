import numpy as np
from typing import Union


def sigmoid(x: Union[np.array, np.float]) -> Union[np.array, np.float]:
    return 1 / (1 + np.exp(-x))
