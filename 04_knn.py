# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
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

#Dividimos los datos en entrenamiento y pruebas
X_train, X_test, Y_train, Y_test=train_test_split(iris['data'],iris['target'],
                                                  train_size=0.80, test_size=0.20)
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

#Establecemos las configuraciones del algoritmo
#empezando por el número de vecinos
knn=KNeighborsClassifier(n_neighbors=6)

#entrenamos al algoritmo con los datos (_train)
knn.fit(X_train,Y_train)

#comprobamos la validez del algortimo
score=knn.score(X_test,Y_test)
print (score)

#Ejemplo de flor con sus características
ret=knn.predict([[1.2,3.4,5.6,1.1]])
#Predicción de tipo de flor
print(iris.target_names[ret])