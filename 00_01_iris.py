from sklearn import datasets
# cargo los datos
iris = datasets.load_iris()
# Saco features
x = iris.data
# saco targets
y = iris.target

from sklearn.model_selection import train_test_split
# divido en train y test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=1, shuffle=False)
from sklearn import tree
# cargo clasificador
classifier = tree.DecisionTreeClassifier()
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
print(prediction)
"""