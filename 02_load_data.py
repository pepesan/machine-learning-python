#importacion de pandas
import pandas
#importación de los modelos de datos de sklearn
from sklearn import datasets

#carga remota desde url del dataset (conjunto de datos)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#etiquetas de los contenidos
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Muestra la forma de los datos
print(dataset.shape)

#carga desde local
dataset=pandas.read_csv("iris.data.csv",names=names)

print(dataset.shape)

#carga desde bibliotecas
dataset = datasets.load_iris()

#conjunto de datos
print(dataset)
#Descripción
print(dataset.DESCR)
#Datos
print(dataset.data)
#Forma de los datos
print(dataset.data.shape)
#Etiquetas aplicables a los resultados
print(dataset.feature_names)
#Resultados esperados del conjunto de datos
print(dataset.target_names)
#Resultados esperados del conjunto de datos
print(dataset.target)