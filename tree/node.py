import numpy as np

from functional import impurity


class Node:

    def __init__(self, X: np.array, y: np.array, t: str, depth: np.int, max_depth: np.int, max_samples: np.int,
                 criterion: str) -> None:
        self.t = t
        self.depth = depth
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.criterion = criterion
        self.X = X
        self.y = y
        self.impurity = impurity(y, criterion)
        if self.t == 'regression':
            self.label = y.std()
        else:
            self.label = np.argmax(np.bincount(y))
        self.feature = None
        self.left = None
        self.right = None
        self.threshold = None
        self.gain = None

    def build(self) -> None:
        if self.depth == self.max_depth or self.X.shape[0] <= self.max_samples:
            return None
        best_feature = None
        best_gain = 0.0
        best_threshold = None
        data = None
        for feature in range(self.X.shape[1]):
            for threshold in self.X[:, feature]:
                X_left, y_left, X_right, y_right = self._split(feature, threshold)
                gain = self.impurity - X_left.shape[0] / self.X.shape[0] * impurity(y_left) + X_right.shape[0] / \
                       self.X.shape[0] * impurity(y_right)
                if gain > best_gain:
                    best_feature = feature
                    best_gain = gain
                    best_threshold = threshold
                    data = (X_left, y_left, X_right, y_right)
        self.feature = best_feature
        self.gain = best_gain
        self.threshold = best_threshold
        if data is not None:
            self.left = Node(data[0], data[1], t=self.t, depth=self.depth + 1, max_depth=self.max_depth,
                             max_samples=self.max_samples, criterion=self.criterion)
            self.right = Node(data[2], data[3], t=self.t, depth=self.depth + 1, max_depth=self.max_depth,
                              max_samples=self.max_samples, criterion=self.criterion)
            self.left.build()
            self.right.build()
        return None

    def _split(self, feature_index: np.int, threshold: np.float) -> (np.array, np.array, np.array, np.array):
        left_indices = self.X[:, feature_index] <= threshold
        right_indices = self.X[:, feature_index] > threshold
        return self.X[left_indices], self.y[left_indices], self.X[right_indices], self.y[right_indices]

    def predict(self, x: np.array) -> np.int:
        if self.feature is not None:
            if x[self.feature] <= self.threshold:
                return self.left.predict(x)
            else:
                return self.right.predict(x)
        return self.label
