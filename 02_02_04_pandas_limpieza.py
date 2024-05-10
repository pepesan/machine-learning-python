import pandas as pd
import numpy as np

df = pd.read_csv("csv/limpia.csv")
print(df)

print(df.head())
print(df.info())
print(df.shape)

datos = df.drop_duplicates()
print('quita duplicados')
print(datos)
print(datos.shape)

datos = df.dropna()
print('quita nulos')
print(datos)
print(datos.shape)

datos = df.fillna(0)
print('rellena nulos')
print(datos)
print(datos.shape)


print('Modificaciones de columnas')
data = {'col1': [1, 2, 3, 4], 'col2': ['a', 'b', 'c', 'd']}
df = pd.DataFrame(data)
print(df)
# Correct approach (modifies original DataFrame using .loc)
df.loc[df['col1'] > 2, 'col2'] = 'X'
print(df)
print('tipos de columnas')
df['col1'] = df['col1'].astype("int")
print(df['col1'].dtype)
print(df['col2'].dtype)


# Create a DataFrame with NaN values
data = {'A': [1, 2, np.nan, 4, 5]}
df = pd.DataFrame(data)

# Replace NaN with the mean of 'A'
df['A'] = df['A'].fillna(df['A'].mean())
print(df)
