import numpy as np 
import pandas as pd 

class SimpleImputer:
    def __init__(self, strategy="mean", fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X):
        is_dataframe = isinstance(X, pd.DataFrame)
        X_df = X if is_dataframe else pd.DataFrame(X)
        self.feature_names_ = X_df.columns

        if self.strategy in ["mean", "median"]:
            numeric_cols = X_df.select_dtypes(include=[np.number])
            if numeric_cols.shape[1] != X_df.shape[1]:
                non_numeric = set(X_df.columns) - set(numeric_cols.columns)
                raise ValueError(
                    f"Strategy '{self.strategy}' cannot be applied to non-numeric columns: {non_numeric}"
                )

            if self.strategy == "mean":
                self.statistics_ = numeric_cols.mean(axis=0, skipna=True).values
            else:  # median
                self.statistics_ = numeric_cols.median(axis=0, skipna=True).values

        elif self.strategy == "constant":
            self.statistics_ = np.array([self.fill_value] * X_df.shape[1])

        elif self.strategy == "most_frequent":
            self.statistics_ = X_df.mode(dropna=True).iloc[0].values

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return self

    def transform(self, X):
        is_dataframe = isinstance(X, pd.DataFrame)
        X_df = X if is_dataframe else pd.DataFrame(X)

        if X_df.shape[1] != len(self.statistics_):
            raise ValueError(
                f"Shape mismatch: fitted on {len(self.statistics_)} features, "
                f"but transform data has {X_df.shape[1]}"
            )

        X_filled = X_df.fillna(pd.Series(self.statistics_, index=self.feature_names_))
        return X_filled if is_dataframe else X_filled.values

    def fit_transform(self, X):
        return self.fit(X).transform(X)
