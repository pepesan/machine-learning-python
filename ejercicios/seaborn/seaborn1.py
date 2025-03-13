if __name__ == '__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Configuración inicial
    sns.set_style("darkgrid")
    # Manejo de warnings
    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # 1. Scatterplot
    print("Ejercicio 1: Scatterplot")
    df = pd.DataFrame({'Edad': np.random.randint(18, 60, 50),
                       'Salario': np.random.randint(20000, 100000, 50)})
    sns.scatterplot(x='Edad', y='Salario', data=df)
    plt.title("Relación entre Edad y Salario")
    plt.show()

    # 2. Lineplot
    print("Ejercicio 2: Lineplot")
    df = pd.DataFrame({'Día': np.arange(1, 11),
                       'Ventas': np.random.randint(100, 500, 10)})
    sns.lineplot(x='Día', y='Ventas', data=df, marker='o')
    plt.title("Ventas Diarias")
    plt.show()

    # 3. Histograma
    print("Ejercicio 3: Histograma")
    df = pd.DataFrame({'Edad': np.random.randint(18, 80, 100)})
    sns.histplot(df['Edad'], bins=10, kde=True)
    plt.title("Distribución de Edades")
    plt.show()

    # 4. Barplot
    print("Ejercicio 4: Barplot")
    df = pd.DataFrame({'Producto': ['A', 'B', 'C', 'D'],
                       'Ventas': [500, 700, 300, 900]})
    sns.barplot(x='Producto', y='Ventas', data=df)
    plt.title("Ventas por Producto")
    plt.show()

    # 5. Boxplot
    print("Ejercicio 5: Boxplot")
    df = pd.DataFrame({'Salario': np.random.randint(20000, 100000, 50)})
    sns.boxplot(y=df['Salario'])
    plt.title("Distribución de Salarios")
    plt.show()

    # 6. Heatmap
    print("Ejercicio 6: Heatmap")
    df = pd.DataFrame(np.random.rand(5,5), columns=list('ABCDE'))
    sns.heatmap(df, annot=True, cmap="coolwarm")
    plt.title("Mapa de Calor")
    plt.show()

    # 7. Pairplot
    print("Ejercicio 7: Pairplot")
    df = sns.load_dataset("iris")
    sns.pairplot(df, hue="species")
    plt.show()

    # 8. Relplot con diferenciación por género
    print("Ejercicio 8: Relplot")
    df = pd.DataFrame({'Edad': np.random.randint(18, 60, 100),
                       'Salario': np.random.randint(20000, 100000, 100),
                       'Género': np.random.choice(['Hombre', 'Mujer'], 100)})
    sns.relplot(x='Edad', y='Salario', hue='Género', data=df, kind='scatter')
    plt.title("Relación entre Edad y Salario diferenciada por Género")
    plt.show()

    # 9. Catplot
    print("Ejercicio 9: Catplot")
    df = pd.DataFrame({'Categoría': np.random.choice(['A', 'B', 'C'], 100),
                       'Ventas': np.random.randint(100, 1000, 100)})
    sns.catplot(x='Categoría', y='Ventas', data=df, kind='box')
    plt.title("Distribución de Ventas por Categoría")
    plt.show()
