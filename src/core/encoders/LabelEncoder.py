import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        y = pd.Series(y)
        self.classes_ = np.unique(y)  
        return self

    def transform(self, y):
        y = pd.Series(y)
        class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        return y.map(class_to_index).to_numpy()

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return np.array([self.classes_[idx] for idx in y])
