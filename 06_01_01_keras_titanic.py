# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

df = pd.read_csv('csv/titanic_dataset.csv')
# Convertir sexo a categoría
df['sex'] = df['sex'].astype('category')

# Convertir categoría a valores numéricos
df['sex'] = df['sex'].cat.codes

# Eliminar filas con valores NaN
df.dropna(inplace=True)
df = df.drop('name', axis=1)
df = df.drop('ticket', axis=1)
# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('survived', axis=1),
    df['survived'], test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(X_train, y_train, epochs=600, batch_size=128)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
