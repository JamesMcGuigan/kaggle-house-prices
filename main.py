#!/usr/bin/env python3

from src import LeastSquaresCorr

model    = LeastSquaresCorr()
scores   = model.fit()
filename = model.output()

print("scores:", scores)
print("wrote:",  filename)
