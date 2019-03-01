import pandas as pd
from copy import deepcopy
from typing import Union
import sklearn.linear_model
import operator

class Model:
    params_default = {
        "id":      "Id",
        "predict": "SalePrice",
        "fields":  [
            "OverallQual",
            "GrLivArea",
            "GarageCars",   # ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
            "GarageArea",   # ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
            "TotalBsmtSF",  # ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
            "1stFlrSF",
            "FullBath",
            "TotRmsAbvGrd",
            "YearBuilt",
            "YearRemodAdd",
        ]
    }

    def __init__(self,
                 train:  Union[str, pd.core.frame.DataFrame],
                 test:   Union[str, pd.core.frame.DataFrame],
                 **kwargs,
    ):
        self.params = dict(self.params_default, **kwargs)

        if isinstance(train, str): train = pd.read_csv(train)
        if isinstance(test,  str): test  = pd.read_csv(test)
        self.data_raw = {
            "train":    train,
            "test":     test,
            "combined": pd.concat([test, train], sort=False).drop( columns=self.params['predict'])
        }

        self.data = {
            "test": self.to_X( self.data_raw['test']  ),
            "X":    self.to_X( self.data_raw['train'] ),
            "Y":    self.to_Y( self.data_raw['train'] ),
        }

        self.models = {
            "LinearRegression": sklearn.linear_model.LinearRegression()
        }

    def to_X( self, dataframe: pd.core.frame.DataFrame ):
        try:
            dataframe = dataframe.drop( columns=self.params['predict'] )
        except:
            pass
        dataframe = dataframe[ [self.params['id']] + self.params['fields'] ]
        dataframe = dataframe.fillna(0)
        return dataframe

    def to_Y( self, dataframe: pd.core.frame.DataFrame ):
        return dataframe[ self.params['predict'] ]


    def fit( self ) -> None:
        for (name, model) in self.models.items():
            model.fit(self.data['X'], self.data['Y'])

    def score( self ) -> dict:
        scores = {}
        for (name, model) in self.models.items():
            scores[name] = model.score(self.data['X'], self.data['Y'])
            
        scores = dict(sorted(scores.items(), key=operator.itemgetter(0), reverse=True))
        return scores

    def best_model_name( self ) -> str:
        scores = self.score()
        return list(scores.keys())[0]

    def predict( self, data=None, model_name=None ) -> dict:
        if data is None: data = self.data['X']

        predictions = {}
        for (name, model) in self.models.items():
            if name == model_name or model_name is None:
                predictions[name] = model.predict( data )
        return predictions

    def output( self, filename ) -> None:
        model_name   = self.best_model_name()
        predictions  = self.predict( data=self.data['test'], model_name=model_name )[model_name]
        ids          = self.data['test'][ self.params['id'] ]
        submission   = pd.DataFrame({ self.params['predict']: predictions }, index=ids )
        csv_text    = submission.to_csv()
        with open(filename, "w") as file:
            file.write(csv_text)
