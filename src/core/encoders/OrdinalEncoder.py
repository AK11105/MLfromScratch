import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class OrdinalEncoder:
    def __init__(self, categories='auto', handle_unknown='error', unknown_value=None, encoded_missing_value=np.nan):
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value

        self.categories_ = None

    def fit(self, X):
        X = pd.DataFrame(X)
        self.categories_ = []

        if self.categories == 'auto':
            for col in X.columns:
                cats = pd.Series(X[col].dropna().unique()).tolist()
                self.categories_.append(cats)
        else:
            if len(self.categories) != X.shape[1]:
                raise ValueError("Length of categories must match number of features")
            self.categories_ = [list(cats) for cats in self.categories]

        return self

    def transform(self, X):
        if self.categories_ is None:
            raise ValueError("This OrdinalEncoder instance is not fitted yet.")

        X = pd.DataFrame(X)
        X_out = np.empty(X.shape, dtype=float)

        for i, col in enumerate(X.columns):
            mapping = {cat: idx for idx, cat in enumerate(self.categories_[i])}

            def encode_value(val):
                if pd.isna(val):
                    return self.encoded_missing_value
                if val in mapping:
                    return mapping[val]
                else:
                    if self.handle_unknown == 'error':
                        raise ValueError(f"Unknown category {val} in column {i}")
                    elif self.handle_unknown == 'use_encoded_value':
                        return self.unknown_value
                    else:
                        raise ValueError(f"Invalid handle_unknown={self.handle_unknown}")

            X_out[:, i] = X[col].apply(encode_value).to_numpy()

        return X_out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        if self.categories_ is None:
            raise ValueError("This OrdinalEncoder instance is not fitted yet.")

        X = np.array(X)
        X_out = pd.DataFrame(index=range(X.shape[0]), columns=range(X.shape[1]))

        for i in range(X.shape[1]):
            cats = self.categories_[i]

            def decode_value(val):
                if pd.isna(val) or val == self.encoded_missing_value:
                    return np.nan
                if val == self.unknown_value:
                    return None  # preserve "unknown"
                if 0 <= int(val) < len(cats):
                    return cats[int(val)]
                else:
                    return None  # safeguard against invalid indices

            X_out.iloc[:, i] = [decode_value(v) for v in X[:, i]]

        return X_out.to_numpy()
