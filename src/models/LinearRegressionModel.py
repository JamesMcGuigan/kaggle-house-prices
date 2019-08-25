from collections import OrderedDict
from typing import Union, Tuple, Dict

import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
from cytoolz.itertoolz import first
from pandas import Series
from pandas.core.frame import DataFrame
from property_cached import cached_property
from sklearn.linear_model.base import LinearModel
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from ..utils import reset_root_dir


class LinearRegressionModel:
    params_default = {
        "id":       "Id",
        "Y_field":  "SalePrice",
        "train":    "./data/train.csv",
        "test":     "./data/test.csv",
        "features": [],
        "comment":  "",
    }

    def __init__(self,
        train:  Union[str, DataFrame] = None,
        test:   Union[str, DataFrame] = None,
        **kwargs,
    ):
        reset_root_dir()
        self.params_default['output'] = f"./data/submissions/{self.__class__.__name__}.csv"
        self.params = dict(self.params_default, **kwargs)

        if train is  None:          train = self.params['train']
        if test  is  None:          test  = self.params['test']
        if isinstance(train, str):  train = pd.read_csv(train)
        if isinstance(test,  str):  test  = pd.read_csv(test)

        # TODO: k-fold validation
        # NOTE: Unable to explain why training / validation splitting has such a major impact on kaggle test scores,
        #       but minimal effect when applied locally. Maybe a smaller dataset leads to less overfitting.
        #
        # Before training / validation dataset splitting:
        # - Your (Kaggle) submission scored 0.43452, which is an improvement of your previous score of 0.74279. Great job!
        # - Kaggle Rank 4079 / 4339
        #
        # After training / validation dataset splitting:
        # - Your (Kaggle) submission scored 0.20892, which is an improvement of your previous score of 0.43452. Great job!
        # - Kaggle Rank 3751 / 4339
        (train, validate) = train_test_split( train, random_state=0 )
        self.data_raw = {
            "test":     test,
            "train":    train,
            "validate": validate,
            "combined": pd.concat([ test, train, validate ], sort=False),
        }
        self.data = {}
        self.init_data()
        self.fit()


    def init_data( self ):
        self.data = {
            "test":     self.to_model( self.data_raw['test']     ),
            "train":    self.to_model( self.data_raw['train']    ),
            "validate": self.to_model( self.data_raw['validate'] ),
            "combined": pd.concat([ self.data_raw['test'], self.data_raw['train'], self.data_raw['validate'] ], sort=False),
        }
        # BUGFIX: Linear Regression Crashes if provided with non-numeric inputs
        self.data.update({
            "X_test":      self.to_X( self.data['test']     ),
            "X_train":     self.to_X( self.data['train']    ),
            "Y_train":     self.to_Y( self.data['train']    ),
            "X_validate":  self.to_X( self.data['validate'] ),
            "Y_validate":  self.to_Y( self.data['validate'] ),
        })



    ##### Executions #####

    def fit( self ):
        for (name, model) in self.models.items():
            # DEBUG: print( "fit X/Y shape", self.data['X_train'].shape, self.data['Y_train'].shape )
            model.fit(self.data['X_train'], self.data['Y_train'])
        return self


    def predict( self, dataframe: DataFrame = None, model_name: str = None ) -> dict:
        if dataframe is None: dataframe = self.data['X_test']

        predictions = {}
        for (name, model) in self.models.items():
            if name == model_name or model_name is None:
                predictions[name] = model.predict( dataframe )
        return predictions



    ##### Transformations #####

    @cached_property
    def models( self ) -> Dict[str, LinearModel]:
        # noinspection PyUnresolvedReferences
        return {
            "LinearRegression": sklearn.linear_model.LinearRegression()
        }

    @cached_property
    def Y_field( self ) -> str:
        return self.params['Y_field']

    def to_model( self, dataframe: DataFrame ) -> DataFrame:
        return dataframe

    def to_X( self, dataframe: DataFrame ) -> DataFrame:
        dataframe = dataframe.drop( columns=self.params['Y_field'], errors='ignore' )
        dataframe = self.X_features( dataframe )
        # noinspection PyProtectedMember
        dataframe = dataframe._get_numeric_data()  # BUGFIX: Linear Regression Crashes if provided with non-numeric inputs
        dataframe = dataframe.fillna(0)            # allow X_features() to do custom fillna() first
        return dataframe

    def X_features( self, dataframe: DataFrame ) -> DataFrame:
        # Extend in subclasses
        return dataframe

    def to_Y( self, dataframe: DataFrame ) -> Series:
        return dataframe[ self.params['Y_field'] ]


    def to_IDs( self, dataframe: DataFrame ) -> Series:
        return dataframe[ self.params['id'] ]



    ##### Scores #####

    def model_scores( self ) -> OrderedDict:
        scores = {}
        for (name, model) in self.models.items():
            scores[name] = {
                "name":     name,
                "model":    model,
                "class":    self.__class__.__name__,
                "features": self.params['features'],
                "R^2":      self.score_r2(model),
                "RMSLE":    self.score_rmsle(model)
            }

        # Sort by RMSLE score - first() == lowest == best
        scores = OrderedDict( sorted(
            scores.items(),
            key=lambda pair: pair[1]['RMSLE'],
            reverse=False
        ))
        return scores

    def score_r2( self, model ):
        # sklearn.linear_model.LinearRegression() uses R^2 as its default scoring method.
        # - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        # - https://www.investopedia.com/terms/r/r-squared.asp
        # DEBUG: print( "score X/Y shape", self.data['X_validate'].shape, self.data['Y_validate'].shape )
        return model.score( self.data['X_validate'], self.data['Y_validate'] )


    def score_rmsle(self, model):
        # Kaggle (for this competition) uses Root-Mean-Squared-Error (RMSE) between
        # the logarithm of the predicted value and the logarithm of the observed sales price.
        # With 0 being a perfect score.
        # - https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/evaluation
        # - https://www.statisticshowto.datasciencecentral.com/rmse/)
        predicted = np.array( model.predict( self.data['X_validate'] ) )
        observed  = np.array(                self.data['Y_validate']   )

        # BUGFIX: {ValueError} Input contains NaN, infinity or a value too large for dtype('float64').
        filter    = ~np.isnan( predicted ) & np.isfinite( predicted ) & (predicted >= 0) & (observed >= 0)
        predicted = predicted[ filter ]
        observed  = observed[  filter ]

        # Manual Calculation of RMSLE - produces same result
        # - https://medium.com/@viveksrinivasan/how-to-finish-top-10-percentile-in-bike-sharing-demand-competition-in-kaggle-part-2-29e854aaab7d
        # calc = ( np.log1p(observed) - np.log1p(predicted) ) ** 2
        # return np.sqrt( np.mean(calc) )

        # NOTE: this is testing against the validation dataset, whereas Kaggle tests against 50% of train dataset
        return np.sqrt( mean_squared_log_error( observed, predicted ) )


    ##### Output #####

    def execute( self ) -> Tuple[float, str, str, str]:
        filename = self.write()
        logfile  = self.summary()
        return logfile


    def summary(self) -> Tuple[float, str, str, str]:
        best = first( self.model_scores().values() )
        # return ( best['RMSLE'], best )
        return ( best['RMSLE'], best['class'], best['name'], " ".join( best['features'] ) )


    def write(self, filename: str = None) -> str:
        if filename is None: filename = self.params['output']

        best         = first( self.model_scores().values() )
        model_name   = best['name']

        ids          = self.to_IDs( self.data['X_test'] )
        predictions  = self.predict( dataframe=self.data['X_test'], model_name=model_name )[model_name]
        submission   = pd.DataFrame({ self.params['Y_field']: predictions }, index=ids ).round(2)
        csv_text     = submission.to_csv()
        with open(filename, "w") as file:
            file.write(csv_text)
            print( f'Wrote: {filename}' )

        return filename


