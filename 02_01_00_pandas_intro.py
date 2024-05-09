# -*- coding: utf-8 -*-

import os
# importacion de pandas
import pandas as pd

# The inital set of baby names and bith rates
names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']
births = [968, 155, 77, 578, 973]
# Dataset
BabyDataSet = list(zip(names, births))
print("DataSet")
print(BabyDataSet)
# [('Bob', 968), ('Jessica', 155), ('Mary', 77), ('John', 578), ('Mel', 973)]
df = pd.DataFrame(data=BabyDataSet, columns=['Names', 'Births'])
print("DataFrame")
print(df)
print("DataFrame Shape")
print(df.shape)
print("DataFrame Describe")
print(df.describe())
print("DataFrame Describe Shape")
print(df.describe().shape)
# Volcamos el DataFrame a fichero
Location = './csv/births1880.csv'
df.to_csv(Location, index=False, header=True)
print(df)
# Lectura de fichero
# df = pd.read_csv(Location, header=False)
x = pd.read_csv(Location)
# df = pd.read_csv(Location, names=['Names', 'Births'])
# df = pd.read_csv(Location, names=['Nombres', 'Nacimientos'])
print("Datos del DataFrame, cargado desde CSV")
print(df)

Sorted = df.sort_values(['Births'], ascending=False)
print("Imprime la primera fila ordenada por nacimientos")
print(Sorted)
print(Sorted.head(2))

print("Imprime el valor máximo de nacimientos")
print(df['Births'].max())

# Carga de un Dataframe
s = pd.Series([1, 2, 3, 4])
print("DataFrame")
print(s)
print("Shape")
print(s.shape)
# Función describe
print("Describe")
print(s.describe())

print("Carga de diccionario")
d = {'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]}
df = pd.DataFrame(data=d)
print("DataFrame")
print(df)
print(df.shape)
print("Describe")
print(df.describe())

# Cómo cargar un DataFrame desde un Bunch de datos de sklearn
import numpy as np
from sklearn.datasets import load_iris

print("iris")
# save load_iris() sklearn dataset to iris
# if you'd like to check dataset type use: type(load_iris())
# if you'd like to view list of attributes use: dir(load_iris())
iris = load_iris()
print(type(iris))
print(type(iris.data))
print(type(iris.target))
# np.c_ is the numpy concatenate function
# which is used to concat iris['data'] and iris['target'] arrays
# for pandas column argument: concat iris['feature_names'] list
# and string list (in this case one string); you can make this anything you'd like..
# the original dataset would probably call this ['Species']
data1 = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                     columns=iris['feature_names'] + ['target'])

print(type(data1))
print(iris['feature_names'])
print(iris['target'])
print(iris['target'].shape)
print(data1)

