# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#ojito que hay que instalar graphviz en el sistema
from sklearn.tree import export_graphviz
import pydotplus
import collections

#carga del fichero CSV
#balance_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',sep= ',', header= None)
balance_data = pd.read_csv('./csv/balance-scale.data.csv',sep= ',', header= None)


#Explorando el dataset
print ("Dataset Lenght:: ", len(balance_data))
print ("Dataset Shape:: ", balance_data.shape)
"""
La etiqueta es el primer valor
“R” : La balanza se orienta a la derecha
“L” : La balanza se orienta a la izquierda
“B” : La balanza se estabiliza

El resto son características que van del 1 al 5 en números naturales
Que son: peso izq, distancia izq, peso derecha, distancia derecha

"""
#Cogemos los valores para tener características y etiquetas
X = balance_data.values[:, 1:5]
Y = balance_data.values[:,0]

#Dividimos en entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


#Entrenamos el algoritmo con gini

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=2, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
print(clf_gini)


#Entrenamos el algoritmo con entropy
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=2, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


#Predecimos con gini
y_pred_gini = clf_gini.predict(X_test)


#Predecimos con entropy
y_pred_en = clf_entropy.predict(X_test)


#Acierto con gini
print ("Acierto con gini es ", accuracy_score(y_test,y_pred_gini)*100)

#Acierto con entropy
print ("Acierto con entropy es ", accuracy_score(y_test,y_pred_en)*100)

#Salvando a fichero .dot
dotfile = open("./figures/arbol.dot", 'w')
feature_names=["Peso Izq", "Dist Izq", "Peso Der", "Dist. Der"]
dot_data=export_graphviz(clf_gini, out_file = dotfile, feature_names = feature_names)
dotfile.close()

#Salvando a fichero png
def genera_png(tree, feature_names, filepath):
    colors = ('turquoise', 'orange')
    dot_data=export_graphviz(tree, out_file = None, feature_names = feature_names,filled=True,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    edges = collections.defaultdict(list)
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph.write_png(filepath)

genera_png(clf_gini,'./figures/tree.png')
genera_png(clf_entropy,'./figures/tree2.png')