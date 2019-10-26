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
| Initial Exploration of the Dataset | [1_dataset.ipynb](1_dataset.ipynb)                                       | 
| Correlation Charts                 | [2_correlations.ipynb](2_correlations.ipynb)                             | 
| Baseline Linear Regression         | [3_baseline_linear_regression.ipynb](3_baseline_linear_regression.ipynb) | 
| Feature Encoding                   | [4_feature_encoding.ipynb](4_feature_encoding.ipynb)                     |
| Sklearn linear_model Exploration   | [5_sklearn_linear_model.ipynb](5_sklearn_linear_model.ipynb)             |
| Quartic Feature Encoding           | [6_quartic_feature_encoding.ipynb](6_quartic_feature_encoding.ipynb)     |

## High Scores

| Score   | Rank        | Class                        | 
|---------|-------------|------------------------------|
| 0.13014 | 1817 / 4926 | fastai.tabular_learner       |
| 0.15502 | 3074 / 4375 | ARDFeatures()                | 
| 0.17628 | 3470 / 4375 | RidgeFeatures()              | 
| 0.17628 | 3493 / 4432 | LarsCVPolynomial()           |
| 0.18379 |             | RidgeCVNormalizePolynomial() |
| 0.20892 | 3751 / 4339 | LinearRegressionModel()      |
| 0.21785 |             | LarsCVLinear()               |
| 0.80406 |             | FeatureEncoding()            |
| 2.29145 |             | LassoLarsSquared()           |
| 2.29145 |             | SquaredFeatureEncoding()     |
| 2.42903 |             | ElasticNetSquared()          |
| 2.87097 |             | PolynomialFeatureEncoding()  |
