# -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt

import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm, model_selection, tree, preprocessing, metrics
import sklearn.ensemble as ske
#pip install tensorflow
import tensorflow as tf
#pip install git+https://github.com/google/skflow.git
#from tensorflow.contrib import skflow

#Importamos el modelo
# pip install xlrd
titanic_df = pd.read_excel('./csv/titanic3.xls', 'titanic3', index_col=None, na_values=['NA'])
#Hay que fijarse cómo se vectorizan los datos por ejemplo survival y class
"""
survival: Survival (0 = no; 1 = yes)
class: Passenger class (1 = first; 2 = second; 3 = third)
name: Name
sex: Sex
age: Age
sibsp: Number of siblings/spouses aboard
parch: Number of parents/children aboard
ticket: Ticket number
fare: Passenger fare
cabin: Cabin
embarked: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat: Lifeboat (if survived)
body: Body number (if did not survive and body was recovered)
"""

print (titanic_df.head(10))

print("Agrupados por clase")
print(titanic_df.groupby('pclass').mean())

class_sex_grouping = titanic_df.groupby(['pclass','sex']).mean()
print("agrupados por clase y sexo")
print(class_sex_grouping)

class_sex_grouping['survived'].plot.bar()


group_by_age = pd.cut(titanic_df["age"], np.arange(0, 90, 10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['survived'].plot.bar()


print("Imprimimos cuenta de datos disponibles")
print(titanic_df.count())


#Quitamos datos no interesantes
titanic_df = titanic_df.drop(['body','cabin','boat'], axis=1)
titanic_df["home.dest"] = titanic_df["home.dest"].fillna("NA")
titanic_df = titanic_df.dropna()

print("Imprimimos cuenta de datos disponibles tras quitarlos")
print(titanic_df.count())

#Función de preprocesador de datos
def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.sex = le.fit_transform(processed_df.sex)
    processed_df.embarked = le.fit_transform(processed_df.embarked)
    processed_df = processed_df.drop(['name','ticket','home.dest'],axis=1)
    return processed_df

processed_df = preprocess_titanic_df(titanic_df)

#datos sin survived
X = processed_df.drop(['survived'], axis=1).values
#sólo los datos de survived
y = processed_df['survived'].values

#extracción de datos de entrenamiento y pruebas
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

#Creación del algoritmo de arbol de decisiones
clf_dt = tree.DecisionTreeClassifier(max_depth=10)

#Entrenamiento
clf_dt.fit (X_train, y_train)
#Puntuación
print("puntuación")
print(clf_dt.score (X_test, y_test))


shuffle_validator = model_selection.ShuffleSplit(len(X),  test_size=0.2, random_state=0)
def test_classifier(clf):
    scores = model_selection.cross_val_score(clf, X, y, cv=shuffle_validator)
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))
#puntuación tras randomización
print("puntuación random")
print(test_classifier(clf_dt))


clf_rf = ske.RandomForestClassifier(n_estimators=50)
print("puntuación random Forest")
test_classifier(clf_rf)

clf_gb = ske.GradientBoostingClassifier(n_estimators=50)
print("puntuación gradient Boosting")
test_classifier(clf_gb)

eclf = ske.VotingClassifier([('dt', clf_dt), ('rf', clf_rf), ('gb', clf_gb)])
print("puntuación clasificador de votos")
test_classifier(eclf)

"""
tf_clf_dnn = skflow.TensorFlowDNNClassifier(hidden_units=[20, 40, 20], n_classes=2, batch_size=256, steps=1000, learning_rate=0.05)
tf_clf_dnn.fit(X_train, y_train)
tf_clf_dnn.score(X_test, y_test)
"""