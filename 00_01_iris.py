from sklearn import datasets
# cargo los datos
iris = datasets.load_iris()
# Saco features
x = iris.data
print(iris.feature_names)
print(x.shape)
print(x[0])
# saco targets
print(iris.target_names)
y = iris.target
print(y.shape)
print(y[0])
from sklearn.model_selection import train_test_split
# divido en train y test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1, shuffle=True)
from sklearn import tree
# cargo clasificador
classifier = tree.DecisionTreeClassifier(
    max_depth=5
)
# entreno
classifier.fit(x_train, y_train)
# predigo pruebas
predictions = classifier.predict(x_test)
print(predictions)
from sklearn.metrics import accuracy_score
# mido resultados
print(accuracy_score(y_test, predictions))
"""
# predice el último valor
prediction = classifier.predict(x_test[-1:])
print(x_test[0])
print(prediction)
"""

from sklearn.neighbors import NearestCentroid
clf = NearestCentroid()
clf.fit(x_train, y_train)
predictions2 = clf.predict(x_test)
print(accuracy_score(y_test, predictions2))
