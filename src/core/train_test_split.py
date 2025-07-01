import numpy as np 
import pandas as pd

def train_test_split(X, y, test_size, shuffle, stratify, random_state):
    np.random.seed(random_state)
    if stratify:
        classes, vals = np.unique(y, return_inverse=True)
        final_train_indices = []
        final_test_indices = []
        all_indices = np.arange(len(y))
        for cls in classes:
            indices = all_indices[vals == cls]
            if shuffle:
                np.random.shuffle(indices)
            test_set_size = max(1, int(round(len(indices) * test_size)))
            test_indices = indices[:test_set_size]
            train_indices = indices[test_set_size:]
            final_train_indices.extend(train_indices)
            final_test_indices.extend(test_indices)
        return X.iloc[final_train_indices], X.iloc[final_test_indices], y.iloc[final_train_indices], y.iloc[final_test_indices]
    else:
        test_set_size = int(X.shape[0] * test_size)
        indices = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(indices)
        test_indices = indices[:test_set_size]
        train_indices = indices[test_set_size:]
        return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]
