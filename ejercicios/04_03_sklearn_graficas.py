# -*- coding: utf-8 -*

#1.- Crea un nuevo fichero sk.data.py

#2.- Carga las dependencias de sklearn datasets
import matplotlib
import pandas as pd
from sklearn.datasets import load_iris, load_boston
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


# Haciéndolo desde un CSV
"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
print(type(dataset))

#4.- Crea las gráficas de cajas, histograma y scatter de los datos de iris
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
dataset.boxplot()
plt.show()

dataset.hist()
plt.show()

scatter_matrix(dataset, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()
"""
"""
#conjunto de datos
print(dataset)
# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()

"""

# Cargando los datos desde sklearn
"""
#3.- Carga los datos de Iris
iris=load_iris()
print(type(iris))
dataset=iris.data
print(type(dataset))
print(dataset)
# shape
print(dataset.shape)

data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],columns= iris['feature_names'] + ['target'])
print(type(data1))
df_data=data1.drop(labels=["target"],axis=1)
labels=data1['target']
print(df_data.describe())
print(labels)
#4.- Crea las gráficas de cajas, histograma y scatter de los datos de iris
df_data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

df_data.hist()
plt.show()

scatter_matrix(df_data, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()


#5.- Haz lo mismo con los datos de boston


boston_data = load_boston()
df_boston = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
#df_boston['target'] = pd.Series(boston_data.target)
print(df_boston.head())
print(type(df_boston))

#4.- Crea las gráficas de cajas, histograma y scatter de los datos de iris
df_boston.plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False)
plt.show()

df_boston.hist()
plt.show()

scatter_matrix(df_boston, alpha=0.2, figsize=(8, 8), diagonal='kde')
plt.show()

#6.- Si te ves con ganas haz las gráfica con Bokeh

"""
from bokeh.plotting import figure, show, output_file
from bokeh.sampledata.iris import flowers
from sklearn.datasets import  load_iris

iris=load_iris()
print(flowers)
print(iris.keys())
print(type(iris.target))
print(type(flowers['species']))
iris_serie=pd.Series(iris.target)
print(type(iris_serie))
#print(iris.data)
print(iris_serie)
print(flowers['species'])
colormap = {0: 'red', 1: 'green', 2: 'blue'}
colors = [colormap[x] for x in iris_serie]
print(colors)


p = figure(title = "Iris Morphology")
p.xaxis.axis_label = 'Petal Length'
p.yaxis.axis_label = 'Petal Width'

p.circle(flowers["petal_length"], flowers["petal_width"],
         color=colors, fill_alpha=0.2, size=10)

output_file("../figures/iris.html", title="iris.py example")

show(p)
