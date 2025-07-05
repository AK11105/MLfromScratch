import numpy as np 
import pandas as pd 

class KNNRegressor:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None
        
    def fit(self, X_train, y_train):
        self.X = np.asarray(X_train)
        self.y = np.asarray(y_train)

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        test_predictions = []
        for test_sample in range(X_test.shape[0]):
            neighbor_distances = []
            neighbors = []
            for train_sample in range(self.X.shape[0]):
                dist = np.linalg.norm(X_test[test_sample] - self.X[train_sample])
                neighbor_distances.append(dist)

            sorted_indices = np.argsort(neighbor_distances)
            for i in range(self.n_neighbors):
                neighbors.append(self.y[sorted_indices[i]])
            pred = np.mean(neighbors)
            test_predictions.append(pred)
        return np.array(test_predictions)

    def score(self, y_pred, y_test):
        y_pred = np.asarray(y_pred)
        y_test = np.asarray(y_test)
        u = ((y_test - y_pred)**2).sum()
        v = ((y_test - y_test.mean())**2).sum()
        r2 = 1 - u/v
        return r2
        