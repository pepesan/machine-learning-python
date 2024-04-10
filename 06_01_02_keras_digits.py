# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

digits = load_digits()

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.25, random_state=42)

# Normalizar los datos entre 0 y 1
X_train = X_train / X_train.max()
X_test = X_test / X_test.max()

forma = X_train.shape[1]
print(forma)
model = Sequential()
model.add(Flatten(input_shape=(forma,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=128)

print(X_train[0])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Save the model to a file (replace 'my_model.h5' with your desired filename)
model.save('my_model.h5')
print('Model saved successfully!')
