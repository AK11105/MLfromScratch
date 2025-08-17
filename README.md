# MLfromScratch

**Implementing machine learning and deep learning algorithms from scratch using foundational libraries like NumPy/Jax, Pandas.**

## About

**MLfromScratch** is a collection of machine learning and deep learning algorithms implemented from the ground up, with a focus on clarity, mathematical transparency, and educational value. All implementations avoid black-box libraries and instead use NumPy as the foundational engine, ensuring that every step of the algorithm is visible and modifiable.

## Features

- Algorithms implemented **from scratch** for maximum transparency
- Minimal dependency on external libraries (primarily uses NumPy)
- Example test harnesses and baseline datasets for quick experimentation
- Clean module structure for easy extension and learning

## Project Structure

```
MLfromScratch/
├── configs/          # Configuration files (hyperparameters, settings)
├── notebooks/        # Jupyter notebooks for demos, experiments, & examples
├── src/              # Source code for algorithms
├── test/             # Test scripts for model evaluation
├── requirements.txt  # Python dependencies
├── setup.py          # Installation script
└── README.md
```

## Supported Algorithms

- **Linear Models:** 
  - Linear Regression
  - Stochastic Gradient Descent
  - Batch Gradient Descent
  - Mini Batch Stochastic Gradient Descent
- **Neighbors:**
  - KNN Classifier
  - KNN Regressor
  - RNN Classifier
  - RNN Regressor
- **Decomposition**
  - PCA

More models coming soon!

## Supported Functionalities
- **Train Test Split** 
- **Scalers:** 
  - StandardScaler
  - MinMaxScaler
  - MaxAbsScaler
  - RobustScaler
- **Imputers:**
  - SimpleImputer
  - KNNImputer
- **Encoders:**
  - LabelEncoder
  - OrdinalEncoder

## Datasets

- **Regression**: Boston Housing Dataset
- **Classification**: Iris Species Dataset

## Running Tests

You can run the built-in tests for each model using:

```bash
python -m test..
```

- ``: e.g., `linear_models`, `neighbors`
- ``: e.g., `LinearRegression`, `KNNClassifier`

**Examples:**
```bash
python -m test.linear_models.test_linear_regression
python -m test.neighbors.knn_classifier
```