import numpy as np

from base import ClassifierMixin, RegressorMixin
from .node import Node


class DecisionTree:

    def __init__(self, t: str, max_depth: np.int, max_samples: np.int, criterion: str, max_features: str) -> None:
        self.t = t
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.criterion = criterion
        self.root = None
        self.max_features = max_features
        self.features = None

    def fit(self, X: np.array, y: np.array) -> None:
        n_features = X.shape[1]
        max_features = n_features
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        self.features = np.random.choice(n_features, max_features)
        X = X[:, self.features]
        self.root = Node(X, y, self.t, 1, self.max_depth, self.max_samples, self.criterion)
        self.root.build()

    def predict(self, X: np.array) -> np.array:
        X = X[:, self.features]
        return np.array([self.root.predict(x) for x in X])


class DecisionTreeClassifier(DecisionTree, ClassifierMixin):

    def __init__(self, max_depth: np.int = 8, max_samples: np.int = 5, criterion: str = 'gini',
                 max_features: str = 'sqrt') -> None:
        super().__init__('classifier', max_depth, max_samples, criterion, max_features)


class DecisionTreeRegressor(DecisionTree, RegressorMixin):

    def __init__(self, max_depth: np.int = 8, max_samples: np.int = 5, criterion: str = 'mse',
                 max_features: str = 'sqrt') -> None:
        super().__init__('regressor', max_depth, max_samples, criterion, max_features)
