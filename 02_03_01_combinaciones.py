import pandas as pd


# Crear DataFrame de empleados
empleados = pd.DataFrame({
    'ID': [101, 102, 103, 104],
    'Nombre': ['Ana', 'Luis', 'Carlos', 'María'],
    'Departamento': ['Ventas', 'IT', 'RRHH', 'Marketing']
})

# Crear DataFrame de salarios
salarios = pd.DataFrame({
    'ID': [101, 102, 103, 105],
    'Salario': [2500, 3200, 4000, 2800]
})

# Unir los DataFrames por la columna "ID" (similar a inner join)
df_merged = empleados.merge(salarios, on='ID', how='inner')

print("### Unir empleados con salarios (INNER JOIN) ###")
print(df_merged, "\n")

# INNER JOIN (Solo coincidencias en ambas tablas)
df_inner = empleados.merge(salarios, on='ID', how='inner')

# LEFT JOIN (Todos los empleados, aunque no tengan salario)
df_left = empleados.merge(salarios, on='ID', how='left')

# RIGHT JOIN (Todos los salarios, aunque no haya empleado)
df_right = empleados.merge(salarios, on='ID', how='right')

# FULL OUTER JOIN (Todos los empleados y todos los salarios, llenando con NaN donde falte info)
df_outer = empleados.merge(salarios, on='ID', how='outer')

print("### INNER JOIN ###\n", df_inner, "\n")
print("### LEFT JOIN ###\n", df_left, "\n")
print("### RIGHT JOIN ###\n", df_right, "\n")
print("### FULL OUTER JOIN ###\n", df_outer, "\n")


# Crear dos DataFrames con la misma estructura
ventas_enero = pd.DataFrame({'ID': [1, 2], 'Producto': ['Laptop', 'Mouse'], 'Ventas': [5, 20]})
ventas_febrero = pd.DataFrame({'ID': [3, 4], 'Producto': ['Teclado', 'Monitor'], 'Ventas': [10, 7]})

# Concatenar filas (axis=0)
df_concatenado = pd.concat([ventas_enero, ventas_febrero], ignore_index=True)

print("### Concatenación de filas (unir datos nuevos) ###")
print(df_concatenado, "\n")


# Crear dos DataFrames con la misma cantidad de filas pero diferente información
clientes = pd.DataFrame({'ID': [1, 2, 3], 'Nombre': ['Ana', 'Luis', 'Carlos']})
compras = pd.DataFrame({'Total Compras': [200, 500, 800]})

# Concatenar columnas (axis=1)
df_concatenado_columnas = pd.concat([clientes, compras], axis=1)

print("### Concatenación de columnas (agregar info nueva) ###")
print(df_concatenado_columnas, "\n")


# Crear dos DataFrames con índices
df_clientes = pd.DataFrame({'Nombre': ['Ana', 'Luis', 'Carlos']}, index=[101, 102, 103])
df_pedidos = pd.DataFrame({'Pedido': ['Laptop', 'Mouse', 'Teclado']}, index=[101, 102, 104])

# Hacer un join por el índice
df_join = df_clientes.join(df_pedidos, how='left')

print("### Join por índice ###")
print(df_join, "\n")


df_empleados = pd.DataFrame({'ID_Empleado': [1, 2, 3], 'Nombre': ['Ana', 'Luis', 'Carlos']})
df_departamentos = pd.DataFrame({'ID_Depto': [1, 2, 4], 'Departamento': ['Ventas', 'IT', 'RRHH']})

# Unir por columnas con nombres distintos
df_fusionado = df_empleados.merge(df_departamentos, left_on='ID_Empleado', right_on='ID_Depto', how='left')

print("### Merge con columnas de nombres diferentes ###")
print(df_fusionado, "\n")


