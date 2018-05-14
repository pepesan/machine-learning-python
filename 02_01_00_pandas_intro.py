# -*- coding: utf-8 -*-

import os
#importacion de pandas
import pandas as pd


# The inital set of baby names and bith rates
names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]
#Dataset
BabyDataSet = list(zip(names,births))
print("DataSet")
print(BabyDataSet)
# [('Bob', 968), ('Jessica', 155), ('Mary', 77), ('John', 578), ('Mel', 973)]
df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])
print("DataFrame")
print(df)
#veremos las filas y las columnas del DataFrame
Location='./csv/births1880.csv'
df.to_csv(Location,index=False,header=False)

#lectura de fichero
df = pd.read_csv(Location, header=None)
df = pd.read_csv(Location, names=['Names','Births'])
print("Datos del DataFrame, cargado desde CSV")
print(df)

Sorted = df.sort_values(['Births'], ascending=False)
print("Imprime la primera fila ordenada por nacimientos")
print(Sorted.head(1))

print("Imprime el valor máximo de nacimientos")
print(df['Births'].max())




#Carga de un Dataframe
s = pd.Series([1, 2, 3, 4])
print("DataFrame")
print(s)
print("Shape")
print(s.shape)
#Función describe
print("Describe")
print(s.describe())

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
print("DataFrame")
print(df)
print(df.shape)
print("Describe")
print(df.describe())

"""

#Borrado de fichero csv
import os
os.remove(Location)

import matplotlib.pyplot as plt
# Create graph
df['Births'].plot()

# Maximum value in the data set
MaxValue = df['Births'].max()

# Name associated with the maximum value
MaxName = df['Names'][df['Births'] == df['Births'].max()].values

# Text to display on graph
Text = str(MaxValue) + " - " + MaxName

# Add text to graph
plt.annotate(Text, xy=(1, MaxValue), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

print("The most popular name")
df=df[df['Births'] == df['Births'].max()]
#plt.show()
"""