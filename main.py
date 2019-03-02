#!/usr/bin/env python3

import pprint

from src import LeastSquaresCorr

pp = pprint.PrettyPrinter(depth=6)

pp.pprint( LeastSquaresCorr().execute() )
