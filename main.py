#!/usr/bin/env python3

import pprint

from src.models import LinearRegressionModel, FeatureEncoding

pp = pprint.PrettyPrinter(depth=6)

results = sorted([
    LinearRegressionModel().execute(),
    FeatureEncoding(features=['X_feature_exclude','X_feature_year_ages','X_feature_label_encode','X_feature_onehot']).execute(),
    FeatureEncoding(features=['X_feature_exclude','X_feature_year_ages','X_feature_label_encode','X_feature_onehot','X_feature_polynomial'], outfile='./data/submissions/PolynomialFeatures.csv').execute(),
])
for result in results:
    print( result )
