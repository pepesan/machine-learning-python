if __name__ == '__main__':
    import pandas as pd

# 1. Creación de DataFrame
data = {
    'Nombre': ['Ana', 'Luis', 'Carlos', 'María', 'Jorge'],
    'Edad': [23, 34, 45, 29, 40],
    'Ciudad': ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao']
}
df = pd.DataFrame(data)

# 2. Acceso a Datos
print(df.head(3))  # Primeras 3 filas
print(df.tail(1))  # Última fila
print(df.loc[2, 'Edad'])  # Edad de la tercera persona

# 3. Filtrado de Datos
print(df[df['Edad'] > 30])  # Mayores de 30 años
print(df[df['Ciudad'].str.contains('a', case=False)])  # Ciudades con 'a'
print(df[(df['Edad'] >= 25) & (df['Edad'] <= 40)])  # Edad entre 25 y 40

# 4. Manipulación de Datos
df['Mayor de Edad'] = df['Edad'] >= 18  # Nueva columna
print(df)
df['Nombre'] = df['Nombre'].str.upper()  # Convertir a mayúsculas
df['Ciudad'] = df['Ciudad'].replace('Madrid', 'Barcelona')  # Reemplazar valores

# 5. Carga y Escritura de Datos
df.to_csv('datos.csv', index=False)  # Guardar en CSV
df_csv = pd.read_csv('datos.csv')
print(df_csv)
df.to_json('datos.json', orient='records', indent=4)  # Guardar en JSON
df_json = pd.read_json('datos.json')
print(df_json)

# 6. Estadísticas Básicas
print("Edad media:", df['Edad'].mean())
print("Desviación estándar:", df['Edad'].std())
print("Cantidad por ciudad:", df['Ciudad'].value_counts())

# 7. Reordenación y Eliminación de Datos
df.drop(columns=['Mayor de Edad'], inplace=True)  # Eliminar columna
df = df.sort_values(by='Edad')  # Ordenar por edad
df = df[['Nombre', 'Edad', 'Ciudad']]  # Reordenar columnas
print(df)


