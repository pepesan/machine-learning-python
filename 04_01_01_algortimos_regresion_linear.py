# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.externals import joblib

boston = load_boston()
print(boston.keys())
print(boston.data.shape)
#print(boston.DESCR)

#creamos el dataframe
bos=pd.DataFrame(boston.data)
print(bos.head())

bos.columns=boston.feature_names
print(bos.head())

#Aquí es donde están los precios
print(boston.target[:5])

#Los colocamos dentro del dataframe
bos['PRICE']=boston.target

#Para generar X quitamos la columna del precio
X= bos.drop("PRICE",axis=1)
Y=bos['PRICE']

#Creamos el modelo
lm= LinearRegression()
print(lm)

#Creamos las muestras de entrenamiento y pruebas
X_train, X_test,Y_train,  Y_test = train_test_split(X,Y, test_size=0.25
                                                    , random_state=2
                                                    )
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#Entrenamos el Modelo
lm.fit(X_train,Y_train)

score=lm.score(X_test,Y_test)
print("Score Modelo:",score)


#Guardar el modelo para usarlo más adelante
localizacion_modelo="./modelos/modelo_regresion_linear_boston.pkl"
joblib.dump(lm,localizacion_modelo)

#recuperar el modelo guardado anteriormente
lm=joblib.load(localizacion_modelo)
score=lm.score(X_test,Y_test)
print("Score guardado:",score)



plt.scatter(lm.predict(X_train),lm.predict(X_train)- Y_train, c="b",s=40, alpha=0.5)
plt.scatter(lm.predict(X_test),lm.predict(X_test)- Y_test, c="g",s=40,)
plt.hlines(y=0, xmax=50, xmin=0)
plt.title("Diagrama de dispersión de entrenamiento (azul), y pruebas (verde)")
plt.show()