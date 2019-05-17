import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Union
from metrics import mean_squared_error, accuracy


class BaseModel(metaclass=ABCMeta):


    @abstractmethod
    def fit(self, X: np.array, y: np.array) -> None:
        """kek"""


    def raw_predict(self, X: np.array) -> Union[np.array, np.float]:
        return X.dot(self.W)


class RegressorMixin:


    def score(self, X: np.array, y: np.array) -> np.float:
        y_pred = self.raw_predict(X)
        score = mean_squared_error(y, y_pred)
        return score


    def predict(self, X: np.array) -> Union[np.array, np.float]:
        return self.raw_predict(X)


class ClassifierMixin:


    def score(self, X: np.array, y: np.array) -> np.float:
        y_pred = self.raw_predict(X)
        return accuracy(y, y_pred)


    def predict(self, X: np.array) -> Union[np.array, np.float]:
        scores = self.raw_predict(X)
        y_pred = np.round(scores)
        return y_pred
