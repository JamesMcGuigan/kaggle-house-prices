import re

import numpy as np
import pandas as pd
from cytoolz import groupby, curry
from pandas import DataFrame, CategoricalDtype
from pandas.core.common import flatten
from sklearn.preprocessing import LabelEncoder

from .LinearRegressionModel import LinearRegressionModel


class FeatureEncoding( LinearRegressionModel ):
    params_default = LinearRegressionModel.params_default
    params_default.update({
        'X_feature_exclude':    ['id'],
        'X_feature_year_ages':  ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'],
        'X_feature_label_encode':  {
            "N,Y":   [ "CentralAir" ],
            "NA,Po,Fa,TA,Gd,Ex": [
                'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
            ]
        },
        'X_feature_onehot': [
            "MoSold",
            "LandSlope", "MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities",
            "LotConfig", "Neighborhood", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "MasVnrType", "Foundation",
            "BsmtExposure", "Heating", "Functional", "GarageFinish", "Fence", "SaleType", "SaleCondition",
            "MiscFeature", "Electrical", "GarageType", "PavedDrive",
            "Condition1", "Condition2",
            "Exterior1st", "Exterior2nd",
            "BsmtFinType1", "BsmtFinType2",
        ],
        "comment":  "",
    })


    def X_features( self, dataframe: DataFrame ):
        dataframe = self.X_feature_year_ages( dataframe )
        dataframe = self.X_feature_label_encode( dataframe )
        dataframe = self.X_feature_onehot( dataframe )
        dataframe = self.X_feature_exclude( dataframe )

        # Check all remaining columns have been converted to numeric
        # assert len( set(dataframe.columns) - set(dataframe._get_numeric_data().columns) ) == 0
        return dataframe


    def X_feature_exclude( self, dataframe: DataFrame ) -> DataFrame:
        dataframe = dataframe.drop( self.params['X_feature_exclude'], axis=1, errors='ignore' )
        return dataframe


    def X_feature_year_ages(self, dataframe: DataFrame) -> DataFrame:
        # year = datetime.now().year  # this is not the last date
        year = 2010                   # dataframe.YrSold.max()
        for fieldname in self.params['X_feature_year_ages']:
            dataframe[ f"{fieldname}_Age" ] = year - dataframe[fieldname]
            dataframe[ f"{fieldname}_Age" ].fillna( dataframe[ f"{fieldname}_Age" ].dropna().mean(), inplace=True )  # NaN -> average age

        # NOTE: encoding absolute years as relative ages actually reduces the score, so don't remove absolute year
        # self.params['X_feature_exclude'] += self.params['X_feature_year_ages']
        return dataframe


    # DOCS: https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
    # DOCS: https://stackoverflow.com/questions/50092911/how-to-map-categorical-data-to-category-encoders-ordinalencoder-in-python-pandas
    # DOCS: https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9
    def X_feature_label_encode(self, dataframe: DataFrame) -> DataFrame:
        for label_string, fieldnames in self.params['X_feature_label_encode'].items():
            labels         = label_string.split(',')
            category_dtype = CategoricalDtype(categories=labels, ordered=True)

            encoder = LabelEncoder()
            encoder.fit( labels )
            for fieldname in fieldnames:
                # Replace NaN with first label 'NA', encoder.transform() will throw exception on unseen values
                dataframe[fieldname] = dataframe[fieldname].astype( category_dtype )
                dataframe[fieldname].fillna( labels[0], inplace=True )
                dataframe[ f"{fieldname}_Numeric" ] = encoder.transform( dataframe[fieldname] )

        self.params['X_feature_exclude'] += list(flatten( self.params['X_feature_label_encode'].values() ))
        return dataframe


    def X_feature_onehot(self, dataframe: DataFrame) -> DataFrame:
        # fieldgroups[basename] = [ fieldname ]
        fieldgroups = groupby(
            curry(re.sub)('\d+(st|nd|rd)?$')(''),  # basename
            self.params['X_feature_onehot']        # fieldnames
        )
        encodings = {}
        for basename, fieldnames in fieldgroups.items():
            # NOTE: in theory, unique_values should be hardcoded based on data_description.txt
            #       for Kaggle, we can cheat and just take unique_values from self.data['combined']
            # BUGFIX: running to_X() separately on test/train/validate datasets results in column name mismatches
            unique_values  = np.unique( self.data['combined'][ fieldnames ].dropna().values )
            category_dtype = CategoricalDtype( categories=unique_values )

            for fieldname in fieldnames:
                dataframe[fieldname] = dataframe[fieldname].astype(category_dtype)
                onehot               = pd.get_dummies( dataframe[fieldname], prefix=basename, prefix_sep='_' )
                if not basename in encodings:  encodings[basename] = onehot
                else:                          encodings[basename] = onehot & encodings[basename]  # Bitwise addition

        # Add additional onehot columns to dataframe
        for basename, onehot in encodings.items():
            dataframe = dataframe.join( onehot )

        # Mark original categorical columns for exclusion
        self.params['X_feature_exclude'] += self.params['X_feature_onehot']
        return dataframe
