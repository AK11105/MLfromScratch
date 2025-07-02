import numpy as np 
import pandas as pd

def train_test_split(X, y, test_size, shuffle=True, stratify=False, random_state=None):
    np.random.seed(random_state)

    is_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)

    n_samples = len(y)
    indices = np.arange(n_samples)

    if stratify:
        classes, vals = np.unique(y, return_inverse=True)
        final_train_indices = []
        final_test_indices = []
        for cls in np.unique(vals):
            cls_indices = indices[vals == cls]
            if shuffle:
                np.random.shuffle(cls_indices)
            test_count = max(1, int(round(len(cls_indices) * test_size)))
            final_test_indices.extend(cls_indices[:test_count])
            final_train_indices.extend(cls_indices[test_count:])
        train_indices = np.array(final_train_indices)
        test_indices = np.array(final_test_indices)
    else:
        if shuffle:
            np.random.shuffle(indices)
        test_count = int(n_samples * test_size)
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]

    if is_pandas:
        return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]
    else:
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
