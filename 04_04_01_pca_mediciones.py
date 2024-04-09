# -*- coding: utf-8 -*-
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Entrenamiento del SVM
t0 = time.time()
modelo_svm_sin_transformar = SVC()
modelo_svm_sin_transformar.fit(X_train, y_train)
t1 = time.time()

tiempo_sin_transformar = t1 - t0
t0 = time.time()
prediccion_svm_sin_transformar = modelo_svm_sin_transformar.predict(X_test)
t1 = time.time()

tiempo_inferencia_sin_transformar = t1 - t0
accuracy_svm_sin_transformar = accuracy_score(y_test, prediccion_svm_sin_transformar)


t0 = time.time()

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

t1 = time.time()

tiempo_pca_con_transformar = t1 - t0

# Entrenamiento del SVM
t0 = time.time()
modelo_svm_con_transformar = SVC()
modelo_svm_con_transformar.fit(X_train_pca, y_train)
t1 = time.time()

tiempo_con_transformar = t1 - t0

t0 = time.time()
prediccion_svm_con_transformar = modelo_svm_con_transformar.predict(X_test_pca)
t1 = time.time()

tiempo_inferencia_con_transformar = t1 - t0
accuracy_svm_con_transformar = accuracy_score(y_test, prediccion_svm_con_transformar)


print("Tiempo entrenamiento sin transformación:", tiempo_sin_transformar)
print("Tiempo inferencia sin transformación:", tiempo_inferencia_sin_transformar)
print("Tiempo transformacion PCA:", tiempo_pca_con_transformar)
print("Tiempo entrenamiento con transformación:", tiempo_con_transformar)
print("Tiempo inferencia con transformación:", tiempo_inferencia_con_transformar)
print("Accuracy SVM sin transformar:", accuracy_svm_sin_transformar)
print("Accuracy SVM con transformar:", accuracy_svm_con_transformar)

# Medir tiempos de procesado
# 1.- tiempo de aplicacion de pca
# 2.- tiempo de entrenamiento con 30 d
# 3.- tiempo de entrenamiento con 2 d
# 4.- medir acierto del algoritmo de clasificacion con 30 d y 2 d
# 5.- gráfica de 30 dimesiones scatter plot (gráfico de dispersion)

