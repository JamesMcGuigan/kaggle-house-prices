import sklearn.linear_model
from cached_property import cached_property

from .Model import Model


class LeastSquaresCorr(Model):
    params_default = dict(Model.params_default, **{
        "output":   "./output/LeastSquaresCorr.csv",
        "comment":  "LinearRegression",
    })

    @cached_property
    def models( self ):
        return {
            "LinearRegression": sklearn.linear_model.LinearRegression()
        }

    @cached_property
    def X_fields( self ) -> list:
        return [
            "Id",
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
