import numpy as np 
import pandas as pd 

class KNNRegressor:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
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
            for train_sample in range(self.X.shape[0]):
                dist = np.linalg.norm(X_test[test_sample] - self.X[train_sample])
                neighbor_distances.append(dist)

            sorted_indices = np.argsort(neighbor_distances)
            neighbors = [self.y[i] for i in sorted_indices[:self.n_neighbors]]

            if self.weights == 'distance':
                distances = [neighbor_distances[i] for i in sorted_indices[:self.n_neighbors]]
                weights = [1 / (d + 1e-5) for d in distances]
                weighted_sum = np.dot(weights, neighbors)
                weight_total = np.sum(weights)
                pred = weighted_sum / weight_total
            else:
                pred = np.mean(neighbors)

            test_predictions.append(pred)

        return np.array(test_predictions)

    def score(self, y_pred, y_test):
        y_pred = np.asarray(y_pred)
        y_test = np.asarray(y_test)
        u = ((y_test - y_pred) ** 2).sum()
        v = ((y_test - y_test.mean()) ** 2).sum()
        r2 = 1 - u / v
        return r2
