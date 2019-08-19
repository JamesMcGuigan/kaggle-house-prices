#!/usr/bin/env python3

import pprint

from src import LinearRegressionModel, CorrelationFilterModel

pp = pprint.PrettyPrinter(depth=6)

pp.pprint(LinearRegressionModel().execute())
pp.pprint(CorrelationFilterModel().execute())
