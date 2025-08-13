import numpy as np 

class RobustScaler:
    def __init__(self):
        self.q1 = None
        self.q2 = None
        self.q3 = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.q1 = np.quantile(X, 0.25, axis=0)
        self.q2 = np.quantile(X, 0.5, axis=0)
        self.q3 = np.quantile(X, 0.75, axis=0)

    def transform(self, X):
        X = np.array(X, dtype=float)
        iqr = self.q3 - self.q1
        safe_iqr = np.where(iqr == 0, 1, iqr)
        return (X-self.q2)/safe_iqr

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)