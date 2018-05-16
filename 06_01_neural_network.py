# -*- coding: utf-8 -*-
import sklearn
from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import  MLPClassifier

iris=load_iris()

X=iris.data

Y=iris.target

X_train, X_test, Y_train, Y_test= train_test_split(X,Y)

#5 capas de 6 nodos cada uno repetido el entrenamiento 900 veces
red=MLPClassifier(max_iter=900,hidden_layer_sizes=(5,6))

ret=red.fit(X_train,Y_train)
print(ret)
score=red.score(X_test, Y_test)

print("Score:",score)