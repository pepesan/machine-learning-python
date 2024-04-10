# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

model = Sequential()
# 64 entradas
model.add(Flatten(input_shape=(64,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# (**For your API application**)

# Load the saved model (assuming the file is in the same directory)
loaded_model = load_model('my_model.h5')

# Use the loaded model for predictions in your API

# Example prediction (replace with your actual prediction logic):
new_data = np.array([[0., 0., 0.125, 0.9375, 0.9375, 1.,
                      0.6875, 0., 0., 0.,
                      0.5, 1., 0.6875, 0.1875, 0., 0., 0.,
                      0., 0.8125, 0.5625,
                      0., 0., 0., 0., 0., 0.3125,
                      1., 0.1875, 0.5625, 0.6875,
                      0.1875, 0., 0., 0.625, 0.9375, 0.9375, 1.,
                      1., 0.6875, 0.,
                      0., 0.375, 1., 0.625, 0.4375, 1.,
                      0.3125, 0., 0., 0.,
                      0.1875, 0.25, 0.9375, 0.5, 0., 0., 0.,
                      0., 0.25, 0.9375,
                      0.4375, 0., 0., 0.]
                     ])
new_data = new_data.reshape((1, 64))  # Reshape to (1, 64)
prediction = loaded_model.predict(new_data)
print(prediction)

