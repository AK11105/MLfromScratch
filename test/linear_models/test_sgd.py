import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.core.train_test_split import train_test_split
from src.core.scalers.StandardScaler import StandardScaler
from src.ML.linear_models.StochasticGradientDescent import StochasticGradientDescent

# --- Path Resolution ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "Boston.csv")

print("Resolved CSV Path:", DATA_PATH)
assert os.path.exists(DATA_PATH), f"File not found at: {DATA_PATH}"

df = pd.read_csv(DATA_PATH)

non_feature_columns = ["Unnamed: 0"]
for col in non_feature_columns:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

X = df.drop(columns=["medv"])
y = df["medv"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=False, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd = StochasticGradientDescent(fit_intercept=True)
sgd.fit(X_train_scaled, y_train)
y_pred = sgd.predict(X_test_scaled)
r2 = sgd.score(y_pred, y_test)

print("RMSE score on benchmark Boston Dataset:", r2)
print("Coefficients:", sgd.coeff_)
print("Intercept:", sgd.intercept_)
