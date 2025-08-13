import numpy as np 

class MinMaxScaler:
    def __init__(self):
        self.max = None
        self.min = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.max = np.max(X, axis=0)
        self.min = np.min(X, axis=0)

    def transform(self, X):
        X = np.array(X, dtype=float)
        denom = self.max - self.min
        # Works for both scalars and arrays
        denom = np.where(denom == 0, 1, denom)
        return (X - self.min) / denom

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)