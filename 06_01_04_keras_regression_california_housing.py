# -*- coding: utf-8 -*-
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.datasets.california_housing import load_data

(x_train, y_train), (x_test, y_test) = load_data()

print(x_train.shape)
# (16512, 8)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

model = Sequential()
model.add(Flatten(input_shape=(8,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model.fit(x_train, y_train,
          epochs=2000, batch_size=1280)

# Después del modelo.fit(), evalúa el modelo en el conjunto de prueba
loss, mae = model.evaluate(x_test, y_test)

print("Error (Loss) en el conjunto de prueba:", loss)
print("MAE (Mean Absolute Error) en el conjunto de prueba:", mae)

# Además, puedes usar el modelo entrenado para hacer predicciones y calcular otras métricas personalizadas si lo deseas
predictions = model.predict(x_test)
print(predictions[0])
# Por ejemplo, puedes calcular el error medio absoluto (MAE) manualmente
import numpy as np
mae_manual = np.mean(np.abs(predictions - y_test))
print("MAE manual en el conjunto de prueba:", mae_manual)
print("Primeros 5 resultados de y_test:")
print(y_test[:5])

model.save('california_housing_model.h5')

new_house = x_test[0]
prediction = model.predict(new_house.reshape(1, 8))
print('Precio predicho:', prediction)
