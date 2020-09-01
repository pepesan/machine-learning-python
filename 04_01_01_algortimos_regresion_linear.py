# -*- coding: utf-8 -*-
# pip instalador de bibliotecas python
# $ pip install pandas matplotlib sklearn
# biblioteca de manejo de datos
import pandas as pd
# biblioteca de presentación de datos
import matplotlib.pyplot as plt
# biblioteca de ML
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from joblib import dump, load

# Boston son datos de casas
# Aqui cargo el dataset de datos originales
# gracias a la biblioteca de scikit learn que provee este dataset
boston = load_boston()
# keys devuelve las claves de los datos almacenados
# son 13 medidas de los bultos
# 506 casos registrados
print(boston.keys())
print(boston.data.shape)
print(boston.DESCR)

#creamos el dataframe
bos=pd.DataFrame(boston.data)
print(bos.head())

bos.columns=boston.feature_names
print(bos.head())

#Aquí es donde están los precios
print(boston.target[:5])

#Los colocamos dentro del dataframe
# Aquí disponemos de los precios para cada casa del dataset
bos['PRICE']=boston.target

#Para generar X quitamos la columna del precio
X= bos.drop("PRICE",axis=1)
Y=bos['PRICE']

#Creamos el modelo
lm= LinearRegression()
print(lm)

#Creamos las muestras de entrenamiento y pruebas
# aquí usamos el método train_test_split para dividir los datos en pruebas y fit
# en este caso dividimos un 25% (0.25) para pruebas con el parametro test_size
# el parámetro random_state permite definir un número que da aleatoriedad
# cuanta más aleatoriedad al dividir los datos mejor resultados normalmente
X_train, X_test,Y_train,  Y_test = train_test_split(X,Y, test_size=0.25
                                                    , random_state=3)
# características = datos iniciales medibles
# target = precio real para dicha casa
# train datos para entrenamiento
# test datos para pruebas
# X = características (CIRM,.....)
# Y = target (precio real)
print("Xtrain" + str(X_train.shape))
print("YTrain" + str(Y_train.shape))
print("Xtest" + str(X_test.shape))
print("Ytest" + str(Y_test.shape))

#Entrenamos el Modelo, sólo usamos los datos de train
# guardamos la datos de test y no los entrenamos con ellos
# de esta manera nos aseguramos que el modelo no los sepa de antemano
lm.fit(X_train,Y_train)

# calculamos la puntuación del modelo generado con los datos de prueba como
# entrada, sabiendo lo que debería salir
score=lm.score(X_test,Y_test)
# Score Modelo: 0.7503116174489232 75% de acierto de margen de media
# acierto es cuanto te acercas del precio real
print("Score Modelo:",score)


#Guardar el modelo para usarlo más adelante
# definimos donde lo guardamos!!! el fichero del modelo
localizacion_modelo="./modelos/modelo_regresion_linear_boston.pkl"
# dump guardar el modelo en fichero
dump(lm, localizacion_modelo)

#recuperar el modelo guardado anteriormente
# leemos de disco el fichero del modelo
lm = load(localizacion_modelo)
score = lm.score(X_test, Y_test)
print("Score guardado:", score)



plt.scatter(lm.predict(X_train),lm.predict(X_train)- Y_train, c="b",s=40, alpha=0.5)
plt.scatter(lm.predict(X_test),lm.predict(X_test)- Y_test, c="g",s=40,)
plt.hlines(y=0, xmax=50, xmin=0)
plt.title("Diagrama de dispersión de entrenamiento (azul), y pruebas (verde)")
plt.show()