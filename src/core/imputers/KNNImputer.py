import numpy as np 

class KNNImputer:
    def __init__(self, n_neighbors=5, missing_values=np.nan, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.missing_values = missing_values
        self.weights = weights
        self.X = None
        self.has_missing = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)  # ensure numeric
        self.X = X
        self.has_missing = np.any(np.isnan(self.X))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)  # ensure numeric
        if not self.has_missing and not np.any(np.isnan(X)):
            return X  # no imputation needed

        X_filled = X.copy()
        for i in range(X.shape[0]):  # each row
            for j in range(X.shape[1]):  # each column
                if np.isnan(X[i, j]):  # check for missing value
                    # rows in training data that have a value for this column
                    valid_rows = ~np.isnan(self.X[:, j])
                    distances = []

                    for n_i in np.where(valid_rows)[0]:
                        if n_i == i and X is self.X:
                            continue  # skip self if imputing training data

                        # only compare features where both rows have values
                        mask = ~np.isnan(X[i, :]) & ~np.isnan(self.X[n_i, :])
                        if not np.any(mask):
                            continue

                        dist = np.linalg.norm(X[i, mask] - self.X[n_i, mask])
                        distances.append((dist, n_i))

                    # sort by distance
                    distances.sort(key=lambda x: x[0])
                    neighbors = distances[:self.n_neighbors]

                    if not neighbors:  # fallback if no neighbors found
                        continue

                    neighbor_vals = [self.X[idx, j] for _, idx in neighbors]

                    if self.weights == "distance":
                        neighbor_dists = [d for d, _ in neighbors]
                        weights = [1 / (d + 1e-5) for d in neighbor_dists]
                        imputed_val = np.dot(weights, neighbor_vals) / np.sum(weights)
                    else:
                        imputed_val = np.mean(neighbor_vals)

                    X_filled[i, j] = imputed_val

        return X_filled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
