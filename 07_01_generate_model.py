from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import pandas as pd
import json

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Save the trained model to a file (using pickle for efficiency)
with open("iris_knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

print("Model saved successfully to iris_knn_model.pkl")
print(iris.target_names)


