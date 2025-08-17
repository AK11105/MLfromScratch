import pandas as pd
import numpy as np
import sys
import os

# --- Resolve project root (3 levels up from this file) ---
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "..", "..", ".."))
sys.path.append(project_root)

print("Project root resolved to:", project_root)

# --- Import your PCA ---
from src.ML.decomposition.PCA import PCA

# --- Load dataset safely ---
csv_path = os.path.join(project_root, "data", "processed", "Boston.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find dataset at {csv_path}")

df = pd.read_csv(csv_path)

X = df.drop(columns=["medv"])
y = df["medv"]

# --- Test PCA ---
def main():
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(Z)

    # Shapes
    print("Original shape:", X.shape)
    print("Reduced shape:", Z.shape)
    print("Reconstructed shape:", X_reconstructed.shape)

    # Reconstruction error
    mse = np.mean((X.to_numpy() - X_reconstructed) ** 2)
    print("Reconstruction MSE:", mse)

    # Explained variance ratio
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Checks
    assert Z.shape[1] == 2, "Reduced data should have 2 components"
    assert X.shape == X_reconstructed.shape, "Reconstructed data shape must match original"
    print("All checks passed ✅")


if __name__ == "__main__":
    main()
