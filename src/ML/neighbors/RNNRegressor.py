import pandas as pd 
import numpy as np 

class RNNRegressor:
    def __init__(self, radius=1.0, weights='uniform'):
        self.radius = radius
        self.weights=weights
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
            indices = []
            for train_sample in range(self.X.shape[0]):
                dist = np.linalg.norm(X_test[test_sample] - self.X[train_sample])
                if dist <= self.radius:
                    neighbor_distances.append(dist)
                    indices.append(train_sample)

            if indices:
                neighbors = [self.y[i] for i in indices]
                if self.weights == 'distance':
                    if 0.0 in neighbor_distances:
                        pred = self.y[indices[neighbor_distances.index(0.0)]]
                    else:
                        weights = [1/(d+1e-5) for d in neighbor_distances]
                        weighted_sum = np.dot(weights, neighbors)
                        weights_total = np.sum(weights)
                        pred = weighted_sum /  weights_total
                else:
                    pred = np.mean(neighbors)
            else:
                # No neighbors found; fallback to global mean
                pred = np.mean(self.y)

            test_predictions.append(pred)
        return np.asarray(test_predictions)

    def score(self, y_pred, y_test):
        y_pred = np.asarray(y_pred)
        y_test = np.asarray(y_test)
        u = ((y_test - y_pred) ** 2).sum()
        v = ((y_test - y_test.mean()) ** 2).sum()
        r2 = 1 - u / v
        return r2
        