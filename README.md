Kaggle Competition - House Prices: Advanced Regression Techniques
-----------------------------------------------------

This is a learning and experimentation project for doing data science analysis for the 
Kaggle - House Prices: Advanced Regression Techniques competition.

[https://www.kaggle.com/c/house-prices-advanced-regression-techniques]()


## Installation
```
./requirements.sh           # Install/Update VirtualEnv
source venv/bin/activate    # Source VirtualEnv
jupyter lab                 # Explore Jupyter Notebooks                  
./main.py                   # Execute Data Pipeline

kaggle competitions submit -c house-prices-advanced-regression-techniques -f data/submissions/LeastSquaresCorr.csv -m "sklearn.linear_model.LinearRegression() on fields: .corr() > 0.5"
```


## High Scores

| Score   | Rank | Method | File   | 
|---------|------|--------|--------| 
| 0.74279 | 4180 | sklearn.linear_model.LinearRegression() on fields: .corr() > 0.5 | [src/models/LeastSquaresCorr.py]()  |


## Data Exploration

| Notebook                                | Analysis                           | 
|-----------------------------------------|------------------------------------| 
| [notebooks/1_dataset.ipynb]()           | Initial Exploration of the Dataset |
| [notebooks/2_correlations.ipynb]()      | Correlation Analysis               |
| [notebooks/3_linear_regression.ipynb]() | Linear Regression Exploration      |


 



