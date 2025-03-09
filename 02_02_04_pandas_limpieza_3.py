import pandas as pd
import numpy as np

# Crear DataFrame con valores nulos
df = pd.DataFrame({
    'Nombre': ['Ana', 'Luis', 'Carlos', None, 'Jorge'],
    'Edad': [23, np.nan, 45, 29, 40],
    'Salario': [2500, 3200, np.nan, 2800, 3700]
})

# Identificar valores nulos
print("### Valores Nulos en el DataFrame ###")
print(df.isnull(), "\n")

# Contar cuántos valores nulos hay en cada columna
print("### Conteo de valores nulos por columna ###")
print(df.isnull().sum(), "\n")

# Eliminar filas con valores nulos
df_sin_nulos = df.dropna()

# Eliminar columnas si todas sus filas son nulas
df_sin_columnas_nulas = df.dropna(axis=1, how='all')

print("### DataFrame sin valores nulos ###")
print(df_sin_nulos, "\n")


# Rellenar valores nulos en 'Edad' con la media de la columna
df.fillna({'Edad': df['Edad'].mean(), 'Salario': 3000, 'Nombre': 'Desconocido'}, inplace=True)

print("### DataFrame con valores nulos rellenados ###")
print(df, "\n")


# Rellenar valores nulos en 'Salario' con un valor fijo (ejemplo: 3000)
df.fillna({'Salario': 3000, 'Nombre': 'Desconocido'}, inplace=True)

print("### DataFrame con valores nulos rellenados en 'Salario' y 'Nombre' ###")
print(df, "\n")

# Crear DataFrame con datos duplicados
df_duplicados = pd.DataFrame({
    'ID': [101, 102, 103, 101, 104],
    'Nombre': ['Ana', 'Luis', 'Carlos', 'Ana', 'Jorge'],
    'Edad': [23, 34, 45, 23, 40]
})

# Identificar duplicados
print("### Filas Duplicadas ###")
print(df_duplicados.duplicated(), "\n")

# Eliminar filas duplicadas
df_sin_duplicados = df_duplicados.drop_duplicates()

print("### DataFrame sin duplicados ###")
print(df_sin_duplicados, "\n")


# Crear DataFrame con datos desordenados
df_texto = pd.DataFrame({
    'Nombre': ['  Ana ', 'LUIS', 'carlos  ', 'MARÍA', 'jorge '],
    'Ciudad': ['Madrid', 'Barcelona', 'Madrid', 'Sevilla', 'Bilbao']
})

# Limpiar espacios en blanco
df_texto['Nombre'] = df_texto['Nombre'].str.strip()

# Convertir a minúsculas
df_texto['Nombre'] = df_texto['Nombre'].str.lower()

# Reemplazar valores incorrectos
df_texto['Ciudad'] = df_texto['Ciudad'].replace({'Madrid': 'MAD', 'Barcelona': 'BCN'})

print("### Datos de texto limpios ###")
print(df_texto, "\n")


# Crear DataFrame con tipos de datos incorrectos
df_tipos = pd.DataFrame({
    'ID': ['101', '102', '103'],
    'Edad': ['23', '34', '45'],
    'Salario': ['2500.50', '3200', '4000']
})

# Convertir a tipos correctos
df_tipos['ID'] = df_tipos['ID'].astype(int)
df_tipos['Edad'] = df_tipos['Edad'].astype(int)
df_tipos['Salario'] = df_tipos['Salario'].astype(float)

print("### DataFrame con tipos de datos corregidos ###")
print(df_tipos.dtypes, "\n")


# Crear DataFrame con un outlier en 'Salario'
df_outliers = pd.DataFrame({
    'Nombre': ['Ana', 'Luis', 'Carlos', 'María', 'Jorge'],
    'Salario': [2500, 3200, 100000, 2800, 3700]  # 100000 es un outlier
})

# Calcular el rango intercuartil (IQR)
Q1 = df_outliers['Salario'].quantile(0.25)
Q3 = df_outliers['Salario'].quantile(0.75)
IQR = Q3 - Q1

# Definir límites para detectar outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtrar valores dentro del rango
df_sin_outliers = df_outliers[(df_outliers['Salario'] >= limite_inferior) & (df_outliers['Salario'] <= limite_superior)]

print("### DataFrame sin outliers ###")
print(df_sin_outliers, "\n")

df_renombrado = df.rename(columns={'Salario': 'Salario Mensual', 'Edad': 'Años'})

print("### DataFrame con nombres de columnas corregidos ###")
print(df_renombrado, "\n")




