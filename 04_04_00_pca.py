# -*- coding: utf-8 -*-
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

mglearn.plots.plot_pca_illustration()
# Carga de datos incial
cancer = load_breast_cancer()
print(cancer.target_names)
# Algoritmo de deteccion de anomalias
pca = PCA(n_components=2)
# Entrenar el algoritmos de reducción de anomalías
pca.fit(cancer.data)
# Datos transformados
transformada = pca.transform(cancer.data)
# 30 dimensiones
print(cancer.data.shape)
# 2 dimensiones
print(transformada.data.shape)


mglearn.discrete_scatter(transformada[:, 0], transformada[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.xlabel("PC1")
plt.ylabel("PC2")
# enseña los datos y su procesado
plt.show()

# Normalización de datos transformados
escala = MinMaxScaler()
# Entrenamiento de la normalización
escala.fit(cancer.data)
# Realizamos la normalización
escalada = escala.transform(cancer.data)
# Entrenamiento de la reducción
pca.fit(escalada)
# aplicación de la reducción
transformada = pca.transform(escalada)
mglearn.discrete_scatter(transformada[:, 0], transformada[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.gca()
plt.xlabel("PC1")
plt.ylabel("PC2")
# enseña los datos y su procesado
plt.show()

# Medir tiempos de procesado
# 1.- tiempo de aplicacion de pca
# 2.- tiempo de entrenamiento con 30 d
# 3.- tiempo de entrenamiento con 2 d
# 4.- medir acierto del algoritmo de clasificacion con 30 d y 2 d
# 5.- gráfica de 30 dimesiones scatter plot (gráfico de dispersion)

