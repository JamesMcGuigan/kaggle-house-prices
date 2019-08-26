from operator import itemgetter
from typing import Dict

import pydash
from property_cached import cached_property
from sklearn import linear_model
from sklearn.linear_model.base import LinearModel

from src.models import LinearRegressionModel, FeatureEncoding, PolynomialFeatureEncoding


# Simplified selection of first-choice models
# A more exhaustive search of sklearn.linear_model requires 3h runtime, but produces the following:
#
# - LarsCV is a good/quick all rounder and best in class for Linear + PolynomialFeatures (and outperforms Ridge)
# - ARDRegression / BayesianRidge are slow but produce the best overall result for FeaturesEncoding
# - Good performers for this dataset: LassoLars, LassoLarsIC, RidgeCV,
# - Worst performer for this dataset: SGDRegressor
class RegularizationModel():
    @cached_property
    def models( self ) -> Dict[str, LinearModel]:
        return {
            "LinearRegression":             linear_model.LinearRegression(),                        # LinearRegression([…])	Ordinary least squares Linear Regression.
            "RidgeCV":                      linear_model.RidgeCV(cv=5),                             # RidgeCV([alphas, …])	Ridge regression with built-in cross-validation.
            "LassoLars":                    linear_model.LassoLars(eps=0.01),                       # LassoLars([alpha, …])	Lasso model fit with Least Angle Regression a.k.a.
            "LassoLarsIC":                  linear_model.LassoLarsIC(eps=0.01),                     # LassoLarsIC([criterion, …])	Lasso model fit with Lars using BIC or AIC for model selection
            "ARDRegression":                linear_model.ARDRegression(),                           #  ARDRegression([n_iter, tol, …])	Bayesian ARD regression.
            "BayesianRidge":                linear_model.BayesianRidge(),                           # BayesianRidge([n_iter, tol, …])	Bayesian ridge regression.
        }

class RegularizationModelLinear(     RegularizationModel, LinearRegressionModel     ): pass
class RegularizationModelFeatures(   RegularizationModel, FeatureEncoding           ): pass
class RegularizationModelPolynomial( RegularizationModel, PolynomialFeatureEncoding ): pass



# A more exhaustive search over options in sklearn.linear_model
class MultiModel():
    @cached_property
    def models( self ) -> Dict[str, LinearModel]:
        return {
            "LinearRegression":             linear_model.LinearRegression(),                        # LinearRegression([…])	Ordinary least squares Linear Regression.

            "ARDRegression":                linear_model.ARDRegression(),                           #  ARDRegression([n_iter, tol, …])	Bayesian ARD regression.
            "BayesianRidge":                linear_model.BayesianRidge(),                           # BayesianRidge([n_iter, tol, …])	Bayesian ridge regression.

            "HuberRegressor":               linear_model.HuberRegressor(),                          # HuberRegressor([epsilon, …])	Linear regression model that is robust to outliers.
            "OrthogonalMatchingPursuitCV":  linear_model.OrthogonalMatchingPursuitCV(cv=5),         # OrthogonalMatchingPursuitCV([…])	Cross-validated Orthogonal Matching Pursuit model (OMP).
            "Perceptron":                   linear_model.Perceptron(max_iter=1000, tol=1e-3),       # Perceptron([penalty, alpha, …])	Read more in the User Guide.
            "RANSACRegressor":              linear_model.RANSACRegressor(),                         # RANSACRegressor([…])	RANSAC (RANdom SAmple Consensus) algorithm.
            "SGDRegressor":                 linear_model.SGDRegressor(max_iter=1000, tol=1e-3),     # SGDRegressor([loss, penalty, …])	Linear model fitted by minimizing a regularized empirical loss with SGD
            "TheilSenRegressor":            linear_model.TheilSenRegressor(),                       # TheilSenRegressor([…])	Theil-Sen Estimator: robust multivariate regression model.
            "PassiveAggressiveRegressor":   linear_model.PassiveAggressiveRegressor(max_iter=1000, tol=1e-3),      # PassiveAggressiveRegressor([C, …])	Passive Aggressive Regressor

            "Lars":                         linear_model.Lars(eps=0.01),                            # Lars([fit_intercept, verbose, …])	Least Angle Regression model a.k.a.
            "LarsCV":                       linear_model.LarsCV(cv=5, eps=0.01),                    # LarsCV([fit_intercept, …])	Cross-validated Least Angle Regression model.
            "Lasso":                        linear_model.Lasso(alpha=1, max_iter=1000),             # Lasso([alpha, fit_intercept, …])	Linear Model trained with L1 prior as regularizer (aka the Lasso)
            "LassoCV":                      linear_model.LassoCV(cv=5),                             # LassoCV([eps, n_alphas, …])	Lasso linear model with iterative fitting along a regularization path.
            "LassoLars":                    linear_model.LassoLars(eps=0.01),                       # LassoLars([alpha, …])	Lasso model fit with Least Angle Regression a.k.a.
            "LassoLarsCV":                  linear_model.LassoLarsCV(cv=5, eps=0.01, max_iter=100), # LassoLarsCV([fit_intercept, …])	Cross-validated Lasso, using the LARS algorithm.
            "LassoLarsIC":                  linear_model.LassoLarsIC(eps=0.01),                     # LassoLarsIC([criterion, …])	Lasso model fit with Lars using BIC or AIC for model selection

            "Ridge":                        linear_model.Ridge(),                                   # Ridge([alpha, fit_intercept, …])	Linear least squares with l2 regularization.
            "RidgeClassifier":              linear_model.RidgeClassifier(),                         # RidgeClassifier([alpha, …])	Classifier using Ridge regression.
            "RidgeClassifierCV":            linear_model.RidgeClassifierCV(cv=5),                   # RidgeClassifierCV([alphas, …])	Ridge classifier with built-in cross-validation.
            "RidgeCV":                      linear_model.RidgeCV(cv=5),                             # RidgeCV([alphas, …])	Ridge regression with built-in cross-validation.
            "SGDClassifier":                linear_model.SGDClassifier(max_iter=1000, tol=1e-3),    # SGDClassifier([loss, penalty, …])	Linear classifiers (SVM, logistic regression, a.o.) with SGD training.

            ### Ignore These
            # "LogisticRegression":           linear_model.LogisticRegression(),                    # LogisticRegression([penalty, …])	Logistic Regression (aka logit, MaxEnt) classifier.
            # "LogisticRegressionCV":         linear_model.LogisticRegressionCV(cv=5),              # LogisticRegressionCV([Cs, …])	Logistic Regression CV (aka logit, MaxEnt) classifier.
            # "MultiTaskLasso":               linear_model.MultiTaskLasso(),                        # MultiTaskLasso([alpha, …])	Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.
            # "MultiTaskElasticNet":          linear_model.MultiTaskElasticNet(),                   # MultiTaskElasticNet([alpha, …])	Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer
            # "MultiTaskLassoCV":             linear_model.MultiTaskLassoCV(cv=5),                  # MultiTaskLassoCV([eps, …])	Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.
            # "MultiTaskElasticNetCV":        linear_model.MultiTaskElasticNetCV(cv=5),             # MultiTaskElasticNetCV([…])	Multi-task L1/L2 ElasticNet with built-in cross-validation.
            # "OrthogonalMatchingPursuit":    linear_model.OrthogonalMatchingPursuit(),             # OrthogonalMatchingPursuit([…])	Orthogonal Matching Pursuit model (OMP)
            # "PassiveAggressiveClassifier":  linear_model.PassiveAggressiveClassifier(),           # PassiveAggressiveClassifier([…])	Passive Aggressive Classifier

            ### Normalization seems to make the score worse!
            # "LinearRegressionNormalize":    linear_model.LinearRegression(normalize=True),          # LinearRegression([…])	Ordinary least squares Linear Regression.
            # "RidgeCVNormalize":             linear_model.RidgeCV(cv=5, normalize=True),             # RidgeCV([alphas, …])	Ridge regression with built-in cross-validation.
            # "LassoLarsNormalize":           linear_model.LassoLars(eps=0.01, normalize=True),       # LassoLars([alpha, …])	Lasso model fit with Least Angle Regression a.k.a.
            # "LassoLarsICNormalize":         linear_model.LassoLarsIC(eps=0.01, normalize=True),     # LassoLarsIC([criterion, …])	Lasso model fit with Lars using BIC or AIC for model selection
            # "ARDRegressionNormalize":       linear_model.ARDRegression(normalize=True),             #  ARDRegression([n_iter, tol, …])	Bayesian ARD regression.
            # "BayesianRidgeNormalize":       linear_model.BayesianRidge(normalize=True),             # BayesianRidge([n_iter, tol, …])	Bayesian ridge regression.
        }

class MultiModelLinear(MultiModel, LinearRegressionModel): pass
class MultiModelFeatures(MultiModel, FeatureEncoding): pass
class MultiModelPolynomial(MultiModel, PolynomialFeatureEncoding): pass


# Best out of: Ridge / Lasso / ElasticNet
class RidgeFeatures(FeatureEncoding):
    @cached_property
    def models( self ) -> Dict[str, LinearModel]:
        return {
            "RidgeCV":          linear_model.RidgeCV(cv=5),            # RidgeCV([alphas, …])	Ridge regression with built-in cross-validation.
        }

class ARDFeatures(FeatureEncoding):
    @cached_property
    def models( self ) -> Dict[str, LinearModel]:
        return {
            "ARDRegression":    linear_model.ARDRegression(),          #  ARDRegression([n_iter, tol, …])	Bayesian ARD regression.
        }

class BayesianRidgeFeatures(FeatureEncoding):
    @cached_property
    def models( self ) -> Dict[str, LinearModel]:
        return {
            "BayesianRidge":    linear_model.BayesianRidge(),          #  ARDRegression([n_iter, tol, …])	Bayesian ARD regression.
        }


# Best score for LinearRegressionModel
class LarsCVLinear(LinearRegressionModel):
    @cached_property
    def models( self ) -> Dict[str, LinearModel]:
        return {
            "LarsCV":           linear_model.LarsCV(cv=5, eps=0.01),   # LarsCV([fit_intercept, …])	Cross-validated Least Angle Regression model.
        }

# Best score for PolynomialFeatureEncoding
class LarsCVPolynomial(PolynomialFeatureEncoding):
    @cached_property
    def models( self ) -> Dict[str, LinearModel]:
        return {
            "LarsCV":           linear_model.LarsCV(cv=5, eps=0.01),   # ARDRegression([n_iter, tol, …])	Bayesian ARD regression.
        }

# New Best score for PolynomialFeatureEncoding
class RidgeCVNormalizePolynomial(PolynomialFeatureEncoding):
    @cached_property
    def models( self ) -> Dict[str, LinearModel]:
        return {
            "RidgeCVNormalizePolynomial": linear_model.RidgeCV(cv=5, normalize=True),   # ARDRegression([n_iter, tol, …])	Bayesian ARD regression.
        }



if __name__ == "__main__":
    results = sorted(pydash.flatten([
        RegularizationModelLinear().model_scores_list(),
        RegularizationModelFeatures().model_scores_list(),
        RegularizationModelPolynomial().model_scores_list(),
    ]), key=itemgetter(0))
    for result in results: print(result)

    # results = sorted(pydash.flatten([
    #     MultiModelLinear().model_scores_list(),
    #     MultiModelFeatures().model_scores_list(),
    #     MultiModelPolynomial().model_scores_list(),
    # ]), key=itemgetter(0))
    # for result in results: print(result)


# These combinations produce the following scores
# $ time python ./src/models/MultiModel.py
#
# (0.1619, 'MultiModelFeatures', 'ARDRegression')
# (0.1619, 'MultiModelFeatures', 'BayesianRidge')
# (0.162, 'MultiModelFeatures', 'RidgeCV')
# (0.1692, 'MultiModelFeatures', 'LassoLars')
# (0.1703, 'MultiModelFeatures', 'Ridge')
# (0.1727, 'MultiModelFeatures', 'LarsCV')
# (0.1727, 'MultiModelFeatures', 'LarsCV')
# (0.1741, 'MultiModelFeatures', 'TheilSenRegressor')
# (0.1758, 'MultiModelFeatures', 'LassoLars')
# (0.1763, 'MultiModelFeatures', 'LassoLarsIC')
# (0.1805, 'MultiModelFeatures', 'Lasso')
# (0.1823, 'MultiModelFeatures', 'LinearRegression')
# (0.1835, 'MultiModelLinear', 'LarsCV')
# (0.1835, 'MultiModelLinear', 'LarsCV')
# (0.1857, 'MultiModelFeatures', 'OrthogonalMatchingPursuitCV')
# (0.189, 'MultiModelPolynomial', 'LarsCV')
# (0.1908, 'MultiModelLinear', 'LassoLarsCV')
# (0.1908, 'MultiModelLinear', 'LassoLarsCV')
# (0.1909, 'MultiModelLinear', 'ARDRegression')
# (0.1909, 'MultiModelPolynomial', 'LassoLarsIC')
# (0.1911, 'MultiModelLinear', 'LassoLarsIC')
# (0.1929, 'MultiModelLinear', 'RidgeCV')
# (0.1938, 'MultiModelLinear', 'RANSACRegressor')
# (0.1939, 'MultiModelLinear', 'LassoLars')
# (0.194, 'MultiModelLinear', 'Ridge')
# (0.1942, 'MultiModelLinear', 'Lasso')
# (0.1942, 'MultiModelLinear', 'LinearRegression')
# (0.1948, 'MultiModelPolynomial', 'LassoLarsCV')
# (0.2039, 'MultiModelFeatures', 'RANSACRegressor')
# (0.2059, 'MultiModelLinear', 'Lars')
# (0.2069, 'MultiModelPolynomial', 'ARDRegression')
# (0.2082, 'MultiModelFeatures', 'HuberRegressor')
# (0.2089, 'MultiModelPolynomial', 'OrthogonalMatchingPursuitCV')
# (0.2098, 'MultiModelLinear', 'BayesianRidge')
# (0.2124, 'MultiModelLinear', 'TheilSenRegressor')
# (0.2128, 'MultiModelFeatures', 'LassoCV')
# (0.2128, 'MultiModelLinear', 'LassoCV')
# (0.2131, 'MultiModelLinear', 'HuberRegressor')
# (0.2174, 'MultiModelFeatures', 'LassoLarsCV')
# (0.2224, 'MultiModelLinear', 'OrthogonalMatchingPursuitCV')
# (0.224, 'MultiModelFeatures', 'LassoLarsCV')
# (0.2242, 'MultiModelFeatures', 'ElasticNetCV')
# (0.2245, 'MultiModelLinear', 'ElasticNetCV')
# (0.2321, 'MultiModelPolynomial', 'LassoLars')
# (0.2321, 'MultiModelPolynomial', 'LassoLars')
# (0.238, 'MultiModelPolynomial', 'Lasso')
# (0.3086, 'MultiModelLinear', 'RidgeClassifier')
# (0.3115, 'MultiModelFeatures', 'RidgeClassifierCV')
# (0.3123, 'MultiModelLinear', 'RidgeClassifierCV')
# (0.3163, 'MultiModelFeatures', 'RidgeClassifier')
# (0.3426, 'MultiModelFeatures', 'PassiveAggressiveRegressor')
# (0.3766, 'MultiModelPolynomial', 'Ridge')
# (0.3773, 'MultiModelPolynomial', 'RidgeClassifier')
# (0.3773, 'MultiModelPolynomial', 'RidgeClassifierCV')
# (0.3818, 'MultiModelPolynomial', 'RidgeCV')
# (0.3929, 'MultiModelPolynomial', 'ElasticNetCV')
# (0.3929, 'MultiModelPolynomial', 'LassoCV')
# (0.3929, 'MultiModelPolynomial', 'LassoCV')
# (0.3939, 'MultiModelPolynomial', 'BayesianRidge')
# (0.4772, 'MultiModelPolynomial', 'Perceptron')
# (0.4973, 'MultiModelPolynomial', 'PassiveAggressiveRegressor')
# (0.5944, 'MultiModelLinear', 'PassiveAggressiveRegressor')
# (0.6264, 'MultiModelPolynomial', 'LinearRegression')
# (0.6705, 'MultiModelPolynomial', 'TheilSenRegressor')
# (0.8323, 'MultiModelLinear', 'Perceptron')
# (0.8382, 'MultiModelFeatures', 'Perceptron')
# (0.9913, 'MultiModelPolynomial', 'RANSACRegressor')
# (1.2163, 'MultiModelPolynomial', 'Lars')
# (9.9689, 'MultiModelPolynomial', 'HuberRegressor')
# (11.167, 'MultiModelFeatures', 'Lars')
# (25.5386, 'MultiModelLinear', 'SGDRegressor')
# (26.4576, 'MultiModelFeatures', 'SGDRegressor')
# (35.2046, 'MultiModelPolynomial', 'SGDRegressor')
#
#
# real    287m25.848s
# user    473m18.367s
# sys     63m18.050s
