if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # 1. Creación de DataFrame
    df = pd.DataFrame({
        'Nombre': ['Ana', 'Luis', 'Carlos', 'María', 'Jorge'],
        'Edad': [23, 34, 45, 29, 40],
        'Ciudad': ['Madrid', 'Barcelona', 'Madrid', 'Sevilla', 'Bilbao']
    })
    print("### DataFrame Inicial ###")
    print(df, "\n")

    # 2. Operaciones Básicas
    print("### Primeras 3 Filas ###")
    print(df.head(3), "\n")
    print("### Información del DataFrame ###")
    print(df.info(), "\n")
    print("### Estadísticas del DataFrame ###")
    print(df.describe(), "\n")

    # 3. Selección de Datos
    print("### Columna Edad ###")
    print(df['Edad'], "\n")
    print("### Personas mayores de 30 años ###")
    print(df[df['Edad'] > 30], "\n")

    # 4. Modificación del DataFrame
    df['Salario'] = [2500, 3200, 4000, 2800, 3700]
    df['Ciudad'] = df['Ciudad'].replace('Madrid', 'MAD')
    print("### DataFrame con Modificaciones ###")
    print(df, "\n")

    # 5. Manejo de Valores Nulos
    df.loc[1, 'Salario'] = np.nan
    df.loc[3, 'Salario'] = np.nan
    print("### DataFrame con Nulos ###")
    print(df, "\n")
    df['Salario'].fillna(df['Salario'].mean(), inplace=True)
    print("### DataFrame con Nulos Rellenados ###")
    print(df, "\n")
    df.dropna(inplace=True)
    print("### DataFrame sin Nulos ###")
    print(df, "\n")

    # 6. Ordenación y Agrupación
    df_sorted = df.sort_values(by='Edad', ascending=False)
    print("### DataFrame Ordenado por Edad ###")
    print(df_sorted, "\n")
    print("### Edad Media por Ciudad ###")
    print(df.groupby('Ciudad')['Edad'].mean(), "\n")

    # 7. Operaciones con Strings
    df['Nombre'] = df['Nombre'].str.upper()
    df['Iniciales'] = df['Nombre'].str[0]
    print("### DataFrame con Nombres en Mayúsculas e Iniciales ###")
    print(df, "\n")

    # 8. Combinación de DataFrames
    departamentos = pd.DataFrame({
        'Nombre': ['ANA', 'LUIS', 'CARLOS', 'MARÍA', 'JORGE'],
        'Departamento': ['Ventas', 'IT', 'RRHH', 'Marketing', 'Finanzas']
    })
    df_merged = df.merge(departamentos, on='Nombre', how='left')
    print("### DataFrame Fusionado con Departamentos ###")
    print(df_merged, "\n")
