import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

class SimpleGBT:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_pred = None

    def _validate_data(self, X, y=None):
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a NumPy array")
        if y is not None and not isinstance(y, np.ndarray):
            raise ValueError("y must be a NumPy array")
        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if np.isnan(X).any():
            raise ValueError("X contains NaN values")
        if y is not None and np.isnan(y).any():
            raise ValueError("y contains NaN values")

    def fit(self, X, y):
        self._validate_data(X, y)
        self.init_pred = np.mean(y)
        pred = np.full_like(y, self.init_pred, dtype=float)
        for _ in range(self.n_estimators):
            residuals = y - pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            update = tree.predict(X)
            pred += self.learning_rate * update

    def predict(self, X):
        self._validate_data(X)

        pred = np.full(X.shape[0], self.init_pred, dtype=float)

        for tree in self.trees:
            update = tree.predict(X)
            pred += self.learning_rate * update

        return pred
