import pandas as pd
import numpy as np


df = pd.DataFrame({
    'ID': [101, 102, 103, 104],
    'Edad': [23, 34, 45, 29],
    'Salario': [2500, 3200, 4000, 2800]
})


# Crear una nueva columna con clasificación de edad
df['Categoría Edad'] = np.where(df['Edad'] >= 30, 'Adulto', 'Joven')

print("### Clasificación de edad con np.where() ###")
print(df[['Edad', 'Categoría Edad']], "\n")


# Definir función para clasificar salarios
def clasificar_salario(salario):
    if salario > 3500:
        return 'Alto'
    elif salario > 2800:
        return 'Medio'
    else:
        return 'Bajo'

# Aplicar la función a la columna 'Salario'
df['Nivel Salario'] = df['Salario'].apply(clasificar_salario)

print("### Clasificación de salarios con apply() ###")
print(df[['Salario', 'Nivel Salario']], "\n")

# Aumentar salario en 15% si es menos a 1500
df['Salario Ajustado'] = df['Salario'].mask(df['Salario'] < 1500, df['Salario'] * 1.15)

print("### Aplicación de condiciones con mask() ###")
print(df[['Salario', 'Salario Ajustado']], "\n")



# Raíz cuadrada de la edad
df['Raíz Edad'] = np.sqrt(df['Edad'])

# Logaritmo del salario
df['Log Salario'] = np.log(df['Salario'])

# Redondeo del salario
df['Salario Redondeado'] = np.round(df['Salario'], decimals=2)

print("### Funciones matemáticas avanzadas con NumPy ###")
print(df, "\n")

