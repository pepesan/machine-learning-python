import pandas as pd
import numpy as np

df = pd.DataFrame([[1, 2.12], [3.356, 4.567], [4.356, 5.567]])
print(df)
print(df.shape)
columnas_a_transformar = ['columna1', 'columna2']  # Specify columns if needed
df.columns = columnas_a_transformar

#def milambda(x):
#    return x-1
df2 = df.map(lambda x: x-1)
print(df2)
print(df2.shape)

df3 = df['columna1'].map(lambda x: x-1)
print(df3)
print(df3.shape)


def resta_uno(x):  # Define a named function for clarity
    if pd.isna(x):
        return np.nan  # Handle missing values appropriately
    else:
        return x - 1


## Primero filtra
df4 = (df['columna1']
       # realiza la transformación
       .map(resta_uno))  # Use map for element-wise operations
# If unnecessary to specify columns, use: df = df.map(resta_uno)

print(df4)
print(df4.shape)


# Filtramos el DataFrame
df_filtrado = df[df['columna1'] > 1]

# Imprimimos el DataFrame filtrado
print(df_filtrado)

# Definimos los valores de corte
valor_corte_columna1 = 2
valor_corte_columna2 = 5

# Filtramos el DataFrame
df_filtrado = df[(df['columna1'] > valor_corte_columna1) & (df['columna2'] > valor_corte_columna2)]

# Imprimimos el DataFrame filtrado
print(df_filtrado)

# Seleccionar filas con valores mayores que 5 en 'A'
print(df.query('columna2 > 5'))


df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'C': [7, 8, 9], 'B': [10, 11, 12]})


# Concatenar verticalmente
print(pd.concat([df1, df2]))

