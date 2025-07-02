MLfromScratch
├── .gitignore
├── configs
├── data
│   ├── processed
│   │   └── Boston.csv
│   └── raw
├── notebooks
│   ├── core
│   │   ├── Scalers.ipynb
│   │   └── TrainTestSplit.ipynb
│   └── linear_models
│       ├── Gradient_Descent.ipynb
│       └── Linear_Regression.ipynb
├── project-structure.md
├── README.md
├── requirements.txt
├── setup.py
├── src
│   ├── core
│   │   ├── scalers
│   │   │   ├── StandardScaler.py
│   │   ├── train_test_split.py
│   │   ├── __init__.py
│   ├── ML
│   │   ├── linear_models
│   │   │   ├── BatchGradientDescent.py
│   │   │   ├── LinearRegression.py
│   │   │   ├── MiniBatchStochasticGradientDescent.py
│   │   │   ├── StochasticGradientDescent.py
│   │   │   ├── __init__.py
│   │   ├── __init__.py
│   ├── __init__.py
└── test
    ├── linear_models
    │   ├── test_bgd.py
    │   ├── test_linear_regression.py
    │   ├── test_mbsgd.py
    │   ├── test_sgd.py
    ├── __init__.py
