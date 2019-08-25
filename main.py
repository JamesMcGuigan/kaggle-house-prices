#!/usr/bin/env python3

import pprint

from src.models import LinearRegressionModel, FeatureEncoding, PolynomialFeatureEncoding

pp = pprint.PrettyPrinter(depth=6)

results = sorted([
    LinearRegressionModel().execute(),
    FeatureEncoding().execute(),
    PolynomialFeatureEncoding().execute(),
])
for result in results:
    print( result )
