import pandas as pd

# ruta al fichero a cargar
ruta = "./iris.data.csv"
# carga del fichero csv
df = pd.read_csv(
    ruta,
    # nombres de las columnas
    names=[
        'p_length', 'p_width',
        's_length', 's_width',
        'target']
)

print(df.shape)
print(df.head)
print(df.describe())

# volcado de datos
ruta_salida = "./iris.data.output.csv"
df.to_csv(ruta_salida, index=False, header=True)


df2 = pd.read_csv(ruta_salida)
print(df2)

