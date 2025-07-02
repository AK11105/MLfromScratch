import numpy as np
import pandas as pd 
import os
import sys

project_root = os.path.abspath(os.path.join(os.getcwd(), "..", "..", ".."))
os.chdir(project_root)
sys.path.append(project_root)
from src.core.train_test_split import train_test_split

class MiniBatchStochasticGradientDescent:
    def __init__(self, alpha=0.001, max_iter=1000, tol=1e-3, fit_intercept = True, early_stopping=False, n_iter_no_change=5, validation_fraction=0.1, shuffle=True, batch_size=32, random_state=None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.shuffle = shuffle
        self.fit_intercept = fit_intercept
        self.random_state = random_state

        self.train_loss_history_ = []
        self.val_loss_history_ = []
        self.intercept_ = None

    def fit(self, X_train, y_train):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.coeff_ = np.zeros(X_train.shape[1])
        best_val_loss = float('inf')
        no_improve_count = 0

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.validation_fraction, shuffle=self.shuffle, stratify=False, random_state=42)
        if self.fit_intercept:
            self.X_mean_ = np.mean(X_train, axis=0)
            self.y_mean_ = np.mean(y_train)
            X_train = X_train - self.X_mean_
            y_train = y_train - self.y_mean_
            X_val = X_val - self.X_mean_
            y_val = y_val - self.y_mean_
        else:
            self.X_mean_ = None
            self.y_mean_ = None

        for epoch in range(self.max_iter):
            if self.shuffle:
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                if isinstance(X_train, pd.DataFrame):
                    X_train = X_train.iloc[indices]
                else:
                    X_train = X_train[indices]

                if isinstance(y_train, pd.Series):
                    y_train = y_train.iloc[indices]
                else:
                    y_train = y_train[indices]

            for start in range(0, X_train.shape[0], self.batch_size):
                end = start + self.batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                y_pred = X_batch @ self.coeff_
                gradient = 2 * X_batch.T @ (y_pred - y_batch) / X_batch.shape[0]
                self.coeff_ -= self.alpha * gradient

            train_pred = X_train @ self.coeff_
            val_pred = X_val @ self.coeff_
            train_loss = np.mean((train_pred - y_train) ** 2)
            val_loss = np.mean((val_pred - y_val) ** 2)

            self.train_loss_history_.append(train_loss)
            self.val_loss_history_.append(val_loss)

            if best_val_loss - val_loss > self.tol:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if self.early_stopping and no_improve_count > self.n_iter_no_change:
                break
        
        if self.fit_intercept:
            self.intercept_ = self.y_mean_ - self.X_mean_ @ self.coeff_


    def predict(self, X_test):
        if self.fit_intercept:
            X_test = X_test - self.X_mean_
            return X_test @ self.coeff_ + self.intercept_
        else:
            return X_test @ self.coeff_

    def score(self, y_pred, y_test):
        mse = np.mean((y_pred - y_test) ** 2)
        rmse = np.sqrt(mse)
        return rmse
