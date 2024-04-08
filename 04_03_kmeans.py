# -*- coding: utf-8 -*-
# Uso de Kmeans para aprendizaje no supervisado
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics

iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Utilizamos Kmeans para encontrar posibles grupos
# n_clusters busca un número de agrupaciones
# hay que jugar con este valor para encontrar el ideal
# max_iter define el número de veces que ejecuta el algoritmo
km = KMeans(n_clusters=3, max_iter=10000)
# Entrenar el modelo
## Sólo usamos los datos (x) no los target (y)
## Estamos en un caso de aprendizaje no supervisado
## No disponemos de la categorías/target
km.fit(X)

predicciones = km.predict(X)
# Vemos las predicciones sobre el set de datos
print(predicciones)
score = metrics.adjusted_rand_score(Y, predicciones)
# Vemos el porcentaje de aciertos respecto a lo esperado
print(score)
