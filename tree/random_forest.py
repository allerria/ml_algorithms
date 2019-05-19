import numpy as np

from base import ClassifierMixin, RegressorMixin
from .decision_tree import DecisionTree


class RandomForest:

    def __init__(self, t: str, criterion: str, n_estimators: np.int = 100, max_features: str = 'sqrt',
                 max_depth: np.int = 9, max_samples: np.int = 1) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.t = t
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_samples = max_samples
        self._estimators = []

    def fit(self, X: np.array, y: np.array) -> None:
        self._estimators = [
            DecisionTree(t=self.t, criterion=self.criterion, max_depth=self.max_depth, max_samples=self.max_samples,
                         max_features=self.max_features) for i in range(self.n_estimators)]
        for i in range(self.n_estimators):
            indices = np.random.choice(range(X.shape[0]), X.shape[0])
            X_i = X[indices, :]
            self._estimators[i].fit(X_i, y)

    def predict(self, X: np.array) -> np.array:
        predicts = np.array([estimator.predict(X) for estimator in self._estimators])
        print(predicts)
        if self.t == 'regressor':
            y_pred = predicts.mean(axis=0)
        else:
            y_pred = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                y_pred[i] = np.argmax(np.bincount(predicts[:, i]))
        return y_pred


class RandomForestRegressor(RandomForest, RegressorMixin):

    def __init__(self, criterion: str = 'mse', n_estimators: np.int = 100, max_features: str = 'sqrt',
                 max_depth: np.int = 9, max_samples: np.int = 1) -> None:
        super().__init__('regressor', criterion, n_estimators, max_features, max_depth, max_samples)


class RandomForestClassifier(RandomForest, ClassifierMixin):

    def __init__(self, criterion: str = 'gini', n_estimators: np.int = 100, max_features: str = 'sqrt',
                 max_depth: np.int = 9, max_samples: np.int = 1) -> None:
        super().__init__('classifier', criterion, n_estimators, max_features, max_depth, max_samples)
