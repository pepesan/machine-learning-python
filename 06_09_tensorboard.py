# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.src.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos
x_train, x_test = x_train / 255.0, x_test / 255.0


# Crear un modelo secuencial
model = Sequential()

# Añadir una capa Flatten para convertir las imágenes en vectores
model.add(Flatten(input_shape=(28, 28)))

# Añadir una capa densa con 128 neuronas
model.add(Dense(128, activation='relu'))

# Añadir una capa densa con 10 neuronas (una para cada clase)
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Compile the model, specifying the optimizer, loss function, and metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Create a TensorBoard callback to log training data
tensorboard_callback = TensorBoard(log_dir='./logs')  # Adjust log directory as needed

# Train the model with TensorBoard logging
model.fit(x_train, y_train, epochs=10, callbacks=[tensorboard_callback])

# (Optional) Evaluate the model on the test set (not shown in this code)
# loss, accuracy = model.evaluate(x_test, y_test)

# Launch TensorBoard (assuming it's installed)
print("Start TensorBoard by running: tensorboard --logdir ./logs")