# !pip install seaborn kagglehub
# dataset de ejemplo
# https://www.kaggle.com/datasets/rohankayan/years-of-experience-and-salary-dataset
import kagglehub
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Download latest version
path = kagglehub.dataset_download("rohankayan/years-of-experience-and-salary-dataset")
complete_path = path + "/Salary_Data.csv"
print("Path to dataset files:", complete_path)


df = pd.read_csv(complete_path)
# Mostrar las primeras filas del dataset
print(df.head())

# Información general del dataset
print(df.info())


# Cargar el dataset
df = pd.read_csv('csv/Salary_Data.csv')

# Mostrar las primeras filas del dataset
print(df.head())

# Información general del dataset
print(df.info())


# Gráfico de dispersión
sns.relplot(x='YearsExperience', y='Salary', data=df, kind='scatter')

# Título del gráfico
plt.title('Relación entre Años de Experiencia y Salario')

# Mostrar el gráfico
plt.show()


# Gráfico de distribución
sns.displot(df['Salary'], kde=True, bins=10)

# Título del gráfico
plt.title('Distribución de Salarios')

# Mostrar el gráfico
plt.show()


# Crear categorías de experiencia
df['Experiencia_Cat'] = pd.cut(df['YearsExperience'], bins=[0, 5, 10, 15, 20, 25], labels=['0-5', '5-10', '10-15', '15-20', '20-25'])

# Gráfico de caja
sns.catplot(x='Experiencia_Cat', y='Salary', data=df, kind='box')

# Título del gráfico
plt.title('Distribución de Salarios por Categoría de Experiencia')

# Mostrar el gráfico
plt.show()

