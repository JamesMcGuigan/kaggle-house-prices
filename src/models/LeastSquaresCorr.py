import sklearn.linear_model
from cached_property import cached_property
from orderedset import OrderedSet

from .LinearModel import LinearModel


class LeastSquaresCorr(LinearModel):
    params_default = dict(LinearModel.params_default, **{
        "output":      "./data/submissions/LeastSquaresCorr.csv",
        "comment":     "LinearRegression",
        "corr_filter": 0.5
    })

    @cached_property
    def models( self ):
        return {
            "LinearRegression": sklearn.linear_model.LinearRegression()
        }

    @cached_property
    def X_fields( self ) -> list:
        Y_corr                  = self.data["train"].corr()[ self.params['Y_field'] ]
        correlated_columns      = Y_corr[ abs(Y_corr) > self.params['corr_filter'] ].sort_values(ascending=False)

        correlated_column_names = [ self.params['id'] ] + list(correlated_columns.index)
        correlated_column_names.remove( self.params['Y_field'] )
        correlated_column_names = list(OrderedSet((correlated_column_names)))
        return correlated_column_names
