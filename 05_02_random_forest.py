# -*- coding: utf-8 -*-
# Pandas is used for data manipulation
import pandas as pd
# Importo numpy para las manipulaciones de datos
import numpy as np
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydotplus
import collections



# Importamos los datos de temperaturas y predicciones
features = pd.read_csv('./csv/temps.csv')
"""
year: 2016 for all data points

month: number for month of the year

day: number for day of the year

week: day of the week as a character string

temp_2: max temperature 2 days prior

temp_1: max temperature 1 day prior

average: historical average max temperature

actual: max temperature measurement

friend: your friend’s prediction, a random number between 20 below the average and 20 above the average
"""
print("Datos en bruto")
print(features.head(5))


# Vectoriza los días de la semana como columnas
features = pd.get_dummies(features)
print("Datos vectorizados")
print(features.head(7))
# Display the first 5 rows of the last 12 columns
#print("Detalle de Datos vectorizados")
#print(features.iloc[:,5:].head(5))


#Obtenemos los datos para el análisis

# Etiquetas desde la columna actual
labels = np.array(features['actual'])
#Quito la columna actual para quedarme con las características

features = features.drop('actual', axis=1)
#Obtengo los nombres de las características
feature_list = list(features.columns)
# Lo convierto a Array
features = np.array(features)


# Obtengo las características y etiquetas de entrenamiento y pruebas
train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels, test_size=0.25, random_state=42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)



# Calculamos las predicciones base  como medias históricas
baseline_preds = test_features[:, feature_list.index('average')]
# Calculamos los errores entre predicciones y etiquetas
baseline_errors = abs(baseline_preds - test_labels)
print('Margen de error medio en grados: ', round(np.mean(baseline_errors), 2))

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, max_depth=3, random_state=42)
# Train the model on training data
rf.fit(train_features, train_labels)


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Error Medio Absoluto:', round(np.mean(errors), 2), 'grados.')


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Acierto:', round(accuracy, 2), '%.')


"""
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

# Pull out one tree from the forest
tree = rf.estimators_[1]
genera_png(tree,feature_list,'./figures/random_tree.png')

"""
