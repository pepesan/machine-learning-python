# -*- coding: utf-8 -*-
from sklearn import linear_model
from sklearn.datasets import fetch_openml

import pandas as pd

lr = linear_model.LinearRegression()

housing = fetch_openml(name="house_prices", as_frame=True)
#print(boston)
#print(boston.DESCR)
#print(boston.target)
print(housing.data.shape)
print(housing.target.shape)
print(housing.feature_names)
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target
y = housing.target
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