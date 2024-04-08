import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el conjunto de datos de diabetes
diabetes = load_diabetes()
X = diabetes.data
print(X)
Y = diabetes.target
print(Y)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42)

# Inicializar y entrenar el modelo de regresión lineal
# Elegir el modelo adecuado para el problema a resolver
model = LinearRegression()
# Realizar el entrenamiento con los datos de entrenamiento
model.fit(X_train, y_train)

# Predecir los valores de la variable dependiente en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio (MSE) en el conjunto de prueba
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio en el conjunto de prueba:", mse)

# Graficar los valores reales vs predichos
plt.scatter(y_test, y_pred)
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Valores reales vs Valores predichos")
plt.show()
