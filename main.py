#!/usr/bin/env python3

import pprint

from src.models import LinearRegressionModel, FeatureEncoding, PolynomialFeatureEncoding, SquaredFeatureEncoding
from src.models import RidgeFeatures, ARDFeatures, LarsCVLinear, LarsCVPolynomial, RidgeCVNormalizePolynomial
from src.models.MultiModel import ElasticNetSquared, LassoLarsSquared

pp = pprint.PrettyPrinter(depth=6)

# results = sorted(pydash.flatten([
#     MultiModelLinear().model_scores_list(),
#     MultiModelFeatures().model_scores_list(),
#     MultiModelPolynomial().model_scores_list(),
# ]))
# for result in results:
#     print( result )


results = sorted([
    # Feature Engineering
    LinearRegressionModel().execute(),       # Your submission scored 0.20892, which is an improvement of your previous score of 0.43452. Great job! | Kaggle Rank 3751 / 4339
    FeatureEncoding().execute(),             # Your submission scored 0.80406, which is not an improvement of your best score (0.20892). Keep trying!
    PolynomialFeatureEncoding().execute(),   # Your submission scored 2.87097, which is not an improvement of your best score (0.20892). Keep trying!
    SquaredFeatureEncoding().execute(),      # Your submission scored 2.29145, which is not an improvement of your best score. Keep trying!

    # Top MultiModels
    LarsCVLinear().execute(),                # Your submission scored 0.21785, which is not an improvement of your best score. Keep trying!
    RidgeCVNormalizePolynomial().execute(),  # Your submission scored 0.18379, which is not an improvement of your best score. Keep trying!
    LarsCVPolynomial().execute(),            # Your submission scored 0.17628, which is an improvement of your previous score of 0.20892. Great job! | Kaggle Rank 3493 / 4432
    RidgeFeatures().execute(),               # Your submission scored 0.17628, which is an improvement of your previous score of 0.20892. Great job! | Kaggle Rank 3470 / 4375
    ARDFeatures().execute(),                 # Your submission scored 0.15502, which is an improvement of your previous score of 0.17628. Great job! | Kaggle Rank 3074 / 4375
    LassoLarsSquared().execute(),            # Your submission scored 2.29145, which is not an improvement of your best score. Keep trying!
    ElasticNetSquared().execute(),           # Your submission scored 2.42903, which is not an improvement of your best score. Keep trying!
])
for result in results:
    print( result )

