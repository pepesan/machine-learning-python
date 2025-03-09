# !pip install seaborn matplotlib pandas numpy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Crear un DataFrame de ejemplo
df = pd.DataFrame({
    'Edad': np.random.randint(18, 60, 50),  # Edades aleatorias entre 18 y 60
    'Salario': np.random.randint(20000, 100000, 50)  # Salarios entre 20k y 100k
})

# Crear un gráfico de dispersión
sns.scatterplot(x='Edad', y='Salario', data=df)

# Mostrar el gráfico
plt.title("Relación entre Edad y Salario")
plt.show()

# Crear datos de ejemplo
df = pd.DataFrame({
    'Día': np.arange(1, 11),  # Días del 1 al 10
    'Ventas': np.random.randint(100, 500, 10)  # Ventas aleatorias
})

# Gráfico de líneas
sns.lineplot(x='Día', y='Ventas', data=df, marker='o')

plt.title("Ventas Diarias")
plt.show()

# Crear un DataFrame con edades aleatorias
df = pd.DataFrame({'Edad': np.random.randint(18, 80, 100)})

# Crear un histograma de edades
sns.histplot(df['Edad'], bins=10, kde=True)

plt.title("Distribución de Edades")
plt.show()

# Crear DataFrame de ventas por producto
df = pd.DataFrame({
    'Producto': ['A', 'B', 'C', 'D'],
    'Ventas': [500, 700, 300, 900]
})

# Gráfico de barras
sns.barplot(x='Producto', y='Ventas', data=df)

plt.title("Ventas por Producto")
plt.show()


# Crear DataFrame con datos de salarios
df = pd.DataFrame({'Salario': np.random.randint(20000, 100000, 50)})

# Gráfico de caja
sns.boxplot(y=df['Salario'])

plt.title("Distribución de Salarios")
plt.show()

# Crear DataFrame de ejemplo
df = pd.DataFrame(np.random.rand(5,5), columns=list('ABCDE'))

# Mapa de calor
sns.heatmap(df, annot=True, cmap="coolwarm")

plt.title("Mapa de Calor")
plt.show()

# Scatter Matrix
sns.set(style="ticks")

df = sns.load_dataset("iris")
print(df)
print(df.shape)
plot = sns.pairplot(df, hue="species")

import matplotlib.pyplot as plt
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Crear DataFrame de ejemplo
df = pd.DataFrame({
    'Edad': np.random.randint(18, 60, 100),
    'Salario': np.random.randint(20000, 100000, 100),
    'Género': np.random.choice(['Hombre', 'Mujer'], 100)
})

# Gráfico relacional con diferenciación por Género
sns.relplot(x='Edad', y='Salario', hue='Género', data=df, kind='scatter')

plt.title("Relación entre Edad y Salario diferenciada por Género")
plt.show()


# Crear DataFrame de ejemplo
df = pd.DataFrame({
    'Categoría': np.random.choice(['A', 'B', 'C'], 100),
    'Ventas': np.random.randint(100, 1000, 100)
})

# Gráfico de caja para ver la distribución de Ventas por Categoría
sns.catplot(x='Categoría', y='Ventas', data=df, kind='box')

plt.title("Distribución de Ventas por Categoría")
plt.show()


# Crear DataFrame de ejemplo con edades
df = pd.DataFrame({'Edad': np.random.randint(18, 80, 100)})

# Gráfico de distribución con KDE activado
sns.displot(df['Edad'], kde=True, bins=15)

plt.title("Distribución de Edades")
plt.show()
