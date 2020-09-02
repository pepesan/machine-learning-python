# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
#print(boston)
#print(boston.DESCR)
#print(boston.target)
print(boston.data.shape)
print(boston.target.shape)
print(boston.feature_names)
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target
y = boston.target
"""
# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, boston.data, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
"""