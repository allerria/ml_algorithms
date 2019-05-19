import numpy as np

from base import BaseModel, ClassifierMixin, RegressorMixin
from .node import Node


class DecisionTree(BaseModel):

    def __init__(self, t: str, max_depth: np.int, max_samples: np.int, criterion: str) -> None:
        self.t = t
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.criterion = criterion
        self.root = None

    def fit(self, X: np.array, y: np.array) -> None:
        self.root = Node(X, y, self.t, 1, self.max_depth, self.max_samples, self.criterion)
        self.root.build()

    def predict(self, X: np.array) -> np.array:
        return np.array([self.root.predict(x) for x in X])


class DecisionTreeClassifier(DecisionTree, ClassifierMixin):

    def __init__(self, max_depth: np.int = 8, max_samples: np.int = 5, criterion: str = 'gini') -> None:
        super().__init__('classifier', max_depth, max_samples, criterion)


class DecisionTreeRegressor(DecisionTree, RegressorMixin):

    def __init__(self, max_depth: np.int = 8, max_samples: np.int = 5, criterion: str = 'mse') -> None:
        super().__init__('regressor', max_depth, max_samples, criterion)
