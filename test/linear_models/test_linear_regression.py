import pandas as pd 
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
os.chdir(project_root)
sys.path.append(project_root)

print("New working directory:", os.getcwd())

from src.core.train_test_split import train_test_split
from src.ML.linear_models.LinearRegression import LinearRegression

df = pd.read_csv("data/processed/Boston.csv")

X = df.drop(columns=["medv"])
y = df["medv"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=False, random_state=42)

linreg = LinearRegression(fit_intercept=False)

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)

r2 = linreg.score(y_pred, y_test)

print("R2 score on benchmark Boston Dataset : ", r2)

coeff = linreg.coeff_
rank = linreg.rank_
singular=linreg.singular_
n = linreg.n_features_in_

print(f"Attributes:", coeff, rank, singular, n)