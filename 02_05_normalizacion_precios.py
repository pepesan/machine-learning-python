# Python code to illustrate
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Lee el archivo CSV en un DataFrame
df = pd.read_csv("csv/ventas.csv")

# Imprime el DataFrame
print(df)

# Assuming 'Precio' is the column containing the price with Euro symbol
df["Precio"] = df["Precio"].str.replace("€", "", regex=True)
df["Total"] = df["Total"].str.replace("€", "", regex=True)
columnas_numericas = ["Precio", "Total"]

scaler = MinMaxScaler()
datos_normalizados = scaler.fit_transform(df[columnas_numericas])

df["Precio_normalizado"] = datos_normalizados[:, 0]
df["Total_normalizado"] = datos_normalizados[:, 1]


import matplotlib.pyplot as plt

plt.scatter(df["Precio"],df["Total"])
plt.title("Datos originales")
plt.show()

plt.scatter(df["Precio_normalizado"], df["Total_normalizado"])
plt.title("Datos normalizados")
plt.show()