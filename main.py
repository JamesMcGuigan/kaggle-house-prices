#!/usr/bin/env python3

import pprint

from src.models import LinearRegressionModel, FeatureEncoding

pp = pprint.PrettyPrinter(depth=6)

pp.pprint( FeatureEncoding().execute() )
pp.pprint( LinearRegressionModel().execute()  )
# pp.pprint( CorrelationFilterModel().execute() )
