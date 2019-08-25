#!/usr/bin/env python3

import pprint

from src.models import LinearRegressionModel, FeatureEncoding

pp = pprint.PrettyPrinter(depth=6)

results = sorted([
    FeatureEncoding().execute(),
    LinearRegressionModel().execute()
])
for result in results:
    print( result )
