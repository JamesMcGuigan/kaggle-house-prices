Kaggle Competition - House Prices: Advanced Regression Techniques
-----------------------------------------------------

This is a learning and experimentation project for doing data science analysis for the 
Kaggle - House Prices: Advanced Regression Techniques competition.

https://www.kaggle.com/c/house-prices-advanced-regression-techniques


## Installation
```
./requirements.sh           # Install/Update VirtualEnv
source venv/bin/activate    # Source VirtualEnv
jupyter lab                 # Explore Jupyter Notebooks                  
./main.py                   # Execute Data Pipeline

kaggle competitions submit -c house-prices-advanced-regression-techniques -f data/submissions/LeastSquaresCorr.csv -m "sklearn.linear_model.LinearRegression() on fields: .corr() > 0.5"
```

## Data Exploration

| Analysis                           | Notebook                                                                                     |  
|------------------------------------|----------------------------------------------------------------------------------------------| 
| Initial Exploration of the Dataset | [notebooks/1_dataset.ipynb](notebooks/1_dataset.ipynb)                                       | 
| Correlation Charts                 | [notebooks/2_correlations.ipynb](notebooks/2_correlations.ipynb)                             | 
| Baseline Linear Regression         | [notebooks/3_baseline_linear_regression.ipynb](notebooks/3_baseline_linear_regression.ipynb) | 


## High Scores

| Date       | Score   | Rank | Method | File | 
|------------|---------|------|--------|------|
| 2019-08-19 | 0.20892 | 3751 | LinearRegression on raw numeric fields + test/validation splitting   | [src/models/LinearRegressionModel.py](src/models/LinearRegressionModel.py) |
| 2019-08-19 | 0.43452 | 4079 | LinearRegression on raw numeric fields                               | [src/models/LinearRegressionModel.py](src/models/LinearRegressionModel.py) | 
| 2019-03-03 | 0.74279 | 4180 | sklearn.linear_model.LinearRegression() on all fields: .corr() > 0.5 | [src/models/LeastSquaresCorr.py](src/models/LeastSquaresCorr.py) |
