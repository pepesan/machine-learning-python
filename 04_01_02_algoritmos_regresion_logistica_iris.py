import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Cargar el conjunto de datos de iris
iris = load_iris()
X = iris.data
print(X)
y = iris.target
print(y)
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42)

# Inicializar y entrenar el modelo de regresión logística
## Escoger el modelo que se ajuste a nuestro problema
## Probamos entre todos los modelos para ver el que mejor funciona
### Colocar los hiperparámetros para jugar con ellos
### Hasta obtener el mejor resultado posible de todas las combinaciones de HP
model = LogisticRegression(max_iter=50)
## Entrenamiento
model.fit(X_train, y_train)

# Predecir las clases en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Imprimir el informe de clasificación
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Graficar las clases reales vs predichas
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
plt.title("Clases reales")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')
plt.title("Clases predichas")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")

plt.tight_layout()
plt.show()
