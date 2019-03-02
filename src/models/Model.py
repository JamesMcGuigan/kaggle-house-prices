import operator
from typing import Union

import pandas as pd
from cached_property import cached_property
from pandas.core.frame import DataFrame
from pandas import Series


class Model:
    params_default = {
        "id":       "Id",
        "Y_field":  "SalePrice",
        "train":    "./data/train.csv",
        "test":     "./data/test.csv",
        "output":   "",
        "comment":  "",
    }

    def __init__(self,
                 train:  Union[str, DataFrame] = None,
                 test:   Union[str, DataFrame] = None,
                 **kwargs,
    ):
        self.params = dict(self.params_default, **kwargs)

        if train is  None:          train = self.params['train']
        if test  is  None:          test  = self.params['test']
        if isinstance(train, str):  train = pd.read_csv(train)
        if isinstance(test,  str):  test  = pd.read_csv(test)
        self.data_raw = {
            "train":    train,
            "test":     test,
            "combined": pd.concat([ test, train ], sort=False),
        }


    @cached_property
    def data( self ):
        return {
            "X_test":       self.to_X( self.data_raw['test'] ),
            "X_train":      self.to_X( self.data_raw['train'] ),
            "Y_train":      self.to_Y( self.data_raw['train'] ),
        }

    @cached_property
    def models( self ):
        return {}

    @cached_property
    def Y_field( self ) -> str:
        return self.params['Y_field']

    @cached_property
    def X_fields( self ) -> list:
        return self.data["test"].columns.values


    def to_X( self, dataframe: DataFrame ) -> DataFrame:
        dataframe = dataframe.drop( columns=self.params['Y_field'], errors='ignore' )
        dataframe = dataframe[ self.X_fields ]
        dataframe = dataframe.fillna(0)
        return dataframe


    def to_Y( self, dataframe: DataFrame ) -> Series:
        return dataframe[ self.params['Y_field'] ]


    def to_IDs( self, dataframe: DataFrame ) -> Series:
        return dataframe[ self.params['id'] ]


    def fit( self ) -> dict:
        for (name, model) in self.models.items():
            model.fit(self.data['X_train'], self.data['Y_train'])
        return self.score()


    def score( self ) -> dict:
        scores = {}
        for (name, model) in self.models.items():
            scores[name] = model.score(self.data['X_train'], self.data['Y_train'])
            
        scores = dict(sorted(scores.items(), key=operator.itemgetter(0), reverse=True))
        return scores


    def best_model_name( self ) -> str:
        scores = self.score()
        return list(scores.keys())[0]


    def predict( self, dataframe: DataFrame = None, model_name: str = None ) -> dict:
        if dataframe is None: dataframe = self.data['X_test']

        predictions = {}
        for (name, model) in self.models.items():
            if name == model_name or model_name is None:
                predictions[name] = model.predict( dataframe )
        return predictions


    def output( self, filename: str = None ) -> str:
        if filename is None: filename = self.params['output']

        model_name   = self.best_model_name()
        predictions  = self.predict( dataframe=self.data['X_test'], model_name=model_name )[model_name]
        ids          = self.to_IDs( self.data['X_test'] )
        submission   = pd.DataFrame({ self.params['Y_field']: predictions }, index=ids )
        csv_text     = submission.to_csv()
        with open(filename, "w") as file:
            file.write(csv_text)

        return filename
