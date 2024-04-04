# Python code to illustrate
import pandas as pd

# Lee el archivo CSV en un DataFrame
df = pd.read_csv("csv/dias.csv")

# Imprime el DataFrame
print(df)


# Crea un diccionario para mapear los días de la semana a los números
dias_map = {
    "Lunes": 1,
    "Martes": 2,
    "Miércoles": 3,
    "Jueves": 4,
    "Viernes": 5,
    "Sábado": 6,
    "Domingo": 7
}

# Crea una nueva columna con el número del día de la semana
df['Numero'] = df['Dia'].map(dias_map)

# Imprime el DataFrame
print(df)