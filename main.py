#!/usr/bin/env python3

from src import Charts, Model

model = Model("./data/train.csv", "./data/test.csv")
model.fit()
scores = model.score()
print("scores:", scores)
model.output("./output/submission.csv")
print("wrote:", "./output/submission.csv")
