import numpy as np 

class MaxAbsScaler:
    def __init__(self):
        self.max = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.max = np.max(np.abs(X), axis=0)

    def transform(self, X):
        X = np.array(X, dtype=float)
        safe_max = np.where(self.max == 0, 1, self.max)
        return X / safe_max

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)