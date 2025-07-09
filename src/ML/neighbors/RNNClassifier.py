import numpy as np
from collections import Counter

class RNNClassifier:
    def __init__(self, radius=1.0, weights='uniform'):
        self.radius = radius
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
            indices = []
            for train_sample in range(self.X.shape[0]):
                dist = np.linalg.norm(X_test[test_sample] - self.X[train_sample])
                if dist <= self.radius:
                    neighbor_distances.append(dist)
                    indices.append(train_sample)

            if indices:
                neighbors = [self.y[i] for i in indices]
                if self.weights == 'distance':
                    if any(np.isclose(d, 0.0) for d in neighbor_distances):
                        zero_idx = next(i for i, d in enumerate(neighbor_distances) if np.isclose(d, 0.0))
                        pred = neighbors[zero_idx]
                    else:
                        weights = [1 / (d + 1e-5) for d in neighbor_distances]
                        class_scores = {}
                        for label, weight in zip(neighbors, weights):
                            class_scores[label] = class_scores.get(label, 0) + weight
                        pred = max(class_scores, key=class_scores.get)
                else:
                    pred = Counter(neighbors).most_common(1)[0][0]
            else:
                # No neighbors found; fallback to global mode
                pred = Counter(self.y).most_common(1)[0][0]

            test_predictions.append(pred)
        return np.asarray(test_predictions)

    def score(self, y_pred, y_true):
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        return (y_pred == y_true).mean()
