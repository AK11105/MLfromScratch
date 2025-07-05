import numpy as np 
import pandas as pd 
from collections import Counter

class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.X = None
        self.y = None
        self.weights = weights

    def fit(self, X_train, y_train):
        self.X = np.asarray(X_train)
        self.y = np.asarray(y_train)

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        test_predictions = []

        for test_sample in range(X_test.shape[0]):
            distances = [np.linalg.norm(X_test[test_sample] - x) for x in self.X]
            sorted_indices = np.argsort(distances)[:self.n_neighbors]
            neighbors = [self.y[i] for i in sorted_indices]

            if self.weights == 'distance':
                weights = [1 / (distances[i] + 1e-5) for i in sorted_indices]
                class_scores = {}
                for label, weight in zip(neighbors, weights):
                    class_scores[label] = class_scores.get(label, 0) + weight
                pred = max(class_scores, key=class_scores.get)
            else:
                pred = Counter(neighbors).most_common(1)[0][0]

            test_predictions.append(pred)

        return np.array(test_predictions)


    def score(self, y_pred, y_true):
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        accuracy = (y_pred == y_true).mean()
        return accuracy
