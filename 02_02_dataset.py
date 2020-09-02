# -*- coding: utf-8 -*-
#importacion de pandas
import pandas as pd

#carga remota desde url del dataset (conjunto de datos)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#etiquetas de los contenidos
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Muestra la forma de los datos
print(dataset.shape)

#carga desde local
dataset = pd.read_csv("./csv/iris.data.csv", names=names)

print(dataset.shape)



#importación de los modelos de datos de sklearn
from sklearn import datasets
#carga desde bibliotecas
dataset = datasets.load_iris()

#conjunto de datos
print(dataset)
#Descripción
print(dataset.DESCR)
#Datos
print(dataset.data)
#Forma de los datos
print(dataset.data.shape)
#Etiquetas aplicables a los resultados
print(dataset.feature_names)
#Resultados esperados del conjunto de datos
print(dataset.target_names)
#Resultados esperados del conjunto de datos
print(dataset.target)



#importación de datos sobre diabetes
#names representa el nombre de las características o datos recogidos en la estructura
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./csv/pima-indians-diabetes.data.csv', names=names)
print("Diabetes")
print(data)
print("imprime los datos medios, standar, minimo")
print(data.describe())



