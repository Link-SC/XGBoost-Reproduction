import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SimpleXGB:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.trees = []
        self.scaler = None

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

    def _preprocess_data(self, X, y=None, fit=False):
        self._validate_data(X, y)
        if fit:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Model has not been fitted yet")
            X = self.scaler.transform(X)
        return X, y

    def fit(self, X, y):
        X, y = self._preprocess_data(X, y, fit=True)
        self.init_pred = np.mean(y)
        pred = np.full_like(y, self.init_pred, dtype=float)

        for _ in range(self.n_estimators):
            # 计算残差
            residuals = y - pred

            # 训练一个新的决策树
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # 更新预测值
            update = tree.predict(X)
            pred += self.learning_rate * update

            # 正则化项
            pred += self.reg_lambda * np.sign(pred)

    def predict(self, X):
        """预测新数据"""
        X, _ = self._preprocess_data(X)
        pred = np.full(X.shape[0], self.init_pred, dtype=float)
        for tree in self.trees:
            update = tree.predict(X)
            pred += self.learning_rate * update
        return pred

# 示例用法
if __name__ == "__main__":
    from sklearn.datasets import make_regression

    # 生成模拟数据
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)

    # 训练模型
    model = SimpleXGB()
    model.fit(X, y)

    # 预测
    y_pred = model.predict(X)
    print(y_pred)
