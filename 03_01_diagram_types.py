# -*- coding: utf-8 -*-
import pandas as pd
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./csv/pima-indians-diabetes.data.csv', names=names)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.pyplot.style.use="default"
#Diagrama de caja
data.boxplot()
plt.show()

#Históriograma
data.hist()
plt.show()

#diagrama de dispersión
from pandas.plotting import scatter_matrix
scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()