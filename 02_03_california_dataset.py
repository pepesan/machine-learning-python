# Referencia: https://medium.com/@basumatary18/implementing-linear-regression-on-california-housing-dataset-378e14e421b7
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

housing = fetch_california_housing(as_frame=True)
housing = housing.frame
housing.head()
print("columnas")
print(housing.columns)

housing.hist(bins=50, figsize=(12, 8))
plt.show()

housing.plot(kind="scatter", x="Longitude", y="Latitude", c="MedHouseVal", cmap="jet", colorbar=True, legend=True,
             sharex=False, figsize=(10, 7), s=housing['Population'] / 100, label="population", alpha=0.7)
plt.show()

attributes = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
              'Latitude', 'Longitude', 'MedHouseVal']
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

housing.plot(kind="scatter", x="MedInc", y="MedHouseVal")
plt.show()

corr = housing.corr()
corr['MedHouseVal'].sort_values(ascending=True)

housing.isna().sum()

print(housing.dtypes)

# pilla features X y target y
X = housing.iloc[:, :-1]
y = housing.iloc[:, -1]
# divide en entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
regression_pipeline = Pipeline([
    ('scaler', scaler),
    ('regressor', LinearRegression())
])
# Entrena
regression_pipeline.fit(X_train, y_train)

# predice
print("predice precio")
y_pred = regression_pipeline.predict(X_test)
print("precio escalado")
print(y_pred[0])
precio_escalado = y_pred[0]
factor_escala = 1 / scaler.scale_[0]
print("factor de escala")
print(factor_escala)
precio_desescalado = precio_escalado * factor_escala
print("Precio desescalado:")
print(precio_desescalado)  # Access the first element of the descaled array

score = r2_score(y_test, y_pred)
# mejor resultado es 1
print("Score")
print(score)
