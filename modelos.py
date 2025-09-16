# modelos.py
import numpy as np

class LinearRegressionFromScratch:
    def __init__(self, theta):
        self.theta = theta
    def predict(self, X):
        if X.shape[1] + 1 == len(self.theta):
            X = np.c_[np.ones((X.shape[0], 1)), X]
        return X.dot(self.theta)

class LinearRegressionCostFromScratch:
    def __init__(self, W, b):
        self.W = W
        self.b = b
    def predict(self, X):
        return X @ self.W + self.b

class LogisticRegressionFromScratch:
    def __init__(self, W, b):
        self.W = W
        self.b = b
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def predict_proba(self, X):
        z = np.dot(X, self.W) + self.b
        return np.hstack([1 - self.sigmoid(z), self.sigmoid(z)])
    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.W) + self.b) >= 0.5).astype(int)
