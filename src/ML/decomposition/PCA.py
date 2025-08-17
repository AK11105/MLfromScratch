import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.n_components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        X = pd.DataFrame(X)

        #Store mean
        self.mean_ = X.mean(axis=0).to_numpy()
        
        # Center data
        X_centred = X - X.mean(axis=0)
        
        # Covariance matrix
        cov_matrix = np.cov(X_centred, rowvar=False)
        
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors (descending order)
        idxs = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idxs]
        eigvecs = eigvecs[:, idxs]
        
        # Number of components
        if self.n_components is None:
            self.n_components_ = len(eigvals)
        else:
            self.n_components_ = self.n_components
        
        # Keep top components
        self.explained_variance_ = eigvals[:self.n_components_]
        self.components_ = eigvecs[:, :self.n_components_]
        
        # Explained variance ratio
        total_var = eigvals.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_centred = X - X.mean(axis=0)
        return np.dot(X_centred, self.components_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        """Reconstruct original data from reduced representation"""
        return np.dot(Z, self.components_.T) + self.mean_