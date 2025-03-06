# -*- coding: utf-8 -*-

#importacion de pandas
import pandas as pd
import sys

print('Python version ' + sys.version)
print('Pandas version: ' + pd.__version__)



# The initial set of baby names and bith rates
names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']
births = [968, 155, 77, 578, 973]
# Dataset
BabyDataSet = list(zip(names, births))
print("DataSet")
print(BabyDataSet)
# [('Bob', 968), ('Jessica', 155), ('Mary', 77), ('John', 578), ('Mel', 973)]
names= ['names', 'births']
df = pd.DataFrame(data=BabyDataSet, columns=names)
print(df)
df.columns = names
print(df['names'])
print(df['births'])
## acceso a varias columnas
print(df[['names','births'] ])

# Our small data set
d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Create dataframe
df = pd.DataFrame(d)
print(df)
print(df.shape)


# Lets change the name of the column
df.columns = ['Rev']
print(df)

## accessing column
print(df['Rev'])
# Lets add a column
df['NewCol'] = 5
print(df)


# Lets modify our new column
df['NewCol'] = df['NewCol'] + 1
print(df)

# Lets add a new column with column data
df['NewCol2'] = df['NewCol'] + 1
print(df)


# We can delete column
del df['NewCol2']
print(df)


# Lets add a couple of columns
df['test'] = 3
df['col'] = df['Rev'] + 1
print(df)


# If we wanted, we could change the name of the index
i = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df.index = i
print(df)


print(df.loc['a'])



# df.loc[inclusive:inclusive]
print(df.loc['a':'d'])


# df.iloc[inclusive:exclusive]
# Note: .iloc is strictly integer position based. It is available from [version 0.11.0] (http://pandas.pydata.org/pandas-docs/stable/whatsnew.html#v0-11-0-april-22-2013)
print(df.iloc[0:3])


print(df['Rev'])


print(df[['Rev', 'test']])


# df.ix[rows,columns]
# replaces the deprecated ix function
#df.ix[0:3,'Rev']
print(df.loc[df.index[0:3],'Rev'])


# replaces the deprecated ix function
#df.ix[5:,'col']
print(df.loc[df.index[5:],'col'])



# replaces the deprecated ix function
#df.ix[:3,['col', 'test']]
print(df.loc[df.index[:3],['col', 'test']])


# Select top N number of records (default = 5)
print(df.head())


# Select bottom N number of records (default = 5)
print(df.tail())


# Crear un DataFrame de ejemplo
data = {
    'ID': [101, 102, 103, 104, 105],
    'Nombre': ['Ana', 'Luis', 'Carlos', 'María', 'Jorge'],
    'Edad': [23, 34, 45, 29, 40],
    'Ciudad': ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao']
}
# Cargar los datos en un DataFrame
df = pd.DataFrame(data)
# Establecer la columna 'ID' como índice
df.set_index('ID', inplace=True)

# Mostrar el DataFrame
print("DataFrame cargado con índice 'ID':")
print(df)

# Seleccionar una fila específica y una columna específica
print("\nFila con ID 102 y columna 'Nombre':")
print(df.loc[102, 'Nombre'])  # Devuelve 'Luis'

# Seleccionar múltiples filas y columnas específicas
print("\nFilas con ID 102 y 104, columnas 'Nombre' y 'Edad':")
print(df.loc[[102, 104], ['Nombre', 'Edad']])

# Seleccionar múltiples filas y columnas específicas
print("\nFilas entre ID 102 y 104, columnas 'Nombre' y 'Edad':")
print(df.loc[102: 104, ['Nombre', 'Edad']])
