import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, fit_intercept):
        self.fit_intercept = fit_intercept
        self.coeff_ = None
        self.rank_ = None
        self.intercept_ = None
        self.n_features_in_ = 0
        self.singular_ = None
        
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]
        self.rank_ = np.linalg.matrix_rank(X)
        self.singular_ = np.linalg.svdvals(X)
        
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centred = X - X_mean
            y_centred = y - y_mean
            self.coeff_ = np.linalg.pinv(X_centred.T @ X_centred) @ X_centred.T @ y_centred
            self.intercept_ = y_mean - X_mean @ self.coeff_
        else:
            self.coeff_ = np.linalg.pinv(X.T @ X) @ X.T @ y
        
    def predict(self, X_test):
        X_test = np.asarray(X_test)
        return X_test @ self.coeff_

        
    def score(self, y_pred, y_test):
        mse = np.mean((y_pred - y_test)**2)
        rmse = np.sqrt(mse)
        return rmse
        