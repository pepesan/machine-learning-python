# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
"""
#Tipo de datos
print(type(iris))
#Datos incluidos
print(iris.keys())
#Nombres de características
print(iris.feature_names)
#Datos de características
print(iris.data)
#Nombres de etiquetas
print(iris.target_names)
#Datos de Etiquetas
print(iris.target)
"""
#Dividimos los datos en entrenamiento y pruebas
X_train, X_test, Y_train, Y_test=train_test_split(iris['data'],iris['target'],
                                                  train_size=0.80, test_size=0.20, random_state=2)
"""
#Datos para entrenamiento
#Forma X_train
print(X_train.shape)
#Forma Y_train
print(Y_train.shape)
#Datos para prueba
#Forma X_test
print(X_test.shape)
#Forma Y_test
print(Y_test.shape)
"""
#Establecemos las configuraciones del algoritmo
#empezando por el número de vecinos
knn=KNeighborsClassifier(n_neighbors=10,weights='distance')

#entrenamos al algoritmo con los datos (_train)
knn.fit(X_train,Y_train)

#comprobamos la validez del algortimo
score=knn.score(X_test,Y_test)
print (score)

#Ejemplo de flor con sus características
ret=knn.predict([[1.2,3.4,5.6,1.1]])
#Predicción de tipo de flor
print(iris.target_names[ret])


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 10

# import some data to play with
iris = datasets.load_iris()

# prepare data
X = iris.data[:, :2]
y = iris.target
h = .02

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])

# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
#algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}
clf.fit(X, y)

# calculate min, max and limits
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# predict class using data and kNN classifier
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)" % (n_neighbors))
plt.show()