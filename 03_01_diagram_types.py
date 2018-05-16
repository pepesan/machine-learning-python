# -*- coding: utf-8 -*-
#Necesitamos instalar el paquete python3-tk con sudo apt-get install
# Import the necessary packages and modules
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Cargamos unos datos de ejemplo
x = np.linspace(0, 10, 100)

# Dibujamos los datos
plt.plot(x, x, label='linear')

# Añadimos una leyenda
plt.legend()

# Mostramos la gráfica
plt.show()


#define una nueva figura
fig = plt.figure()
#Define 111 1 fila, 1 columna y 1 diagrama
ax = fig.add_subplot(111)

#Define las líneas a dibujar mediante puntos definidos en dos ejes
ax.plot([1, 2, 3, 4], [10, 20, 23, 30], color='lightblue', linewidth=3)
#Define los puntos que quiere pintar, 4 en este caso con dos arrays
ax.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26],color='darkgreen', marker='^')
#Define los límites de la gráfica
ax.set_xlim(0.0, 4.5)
#Muestra la grárica
plt.show()



# Inicializa la figura con un tamaño fijo
fig = plt.figure(figsize=(20,10))
#Crea una figura 121 1 fila 2 columnas 1 diagrama
ax1 = fig.add_subplot(121)
#Crea una figura 122 1 fila 2 columnas 2º diagrama
ax2 = fig.add_subplot(122)

# or replace the three lines of code above by the following line:
#fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))

# Gráfico de barras vertical
ax1.bar([1,2,3],[3,4,5])
# Gráfico de barras horizontal
ax2.barh([0.5,1,2.5],[0,1,2])

# Show the plot
plt.show()


"""
Tipos de diagramas básicos
ax.bar()	Vertical rectangles
ax.barh()	Horizontal rectangles
ax.axhline()	Horizontal line across axes
ax.vline()	Vertical line across axes
ax.fill()	Filled polygons
ax.fill_between()	Fill between y-values and 0
ax.stackplot()	Stack plot
"""
# Initialize the plot
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

# Plot the data
ax1.bar([1,2,3],[3,4,5])
ax2.barh([0.5,1,2.5],[0,1,2])
ax2.axhline(0.45)
ax1.axvline(0.65)
ax3.scatter([1,2,3,4],[2,3,4,5])
#Salvar figura
plt.savefig("./figures/demo-figure.png")

from matplotlib.backends.backend_pdf import PdfPages

# Inicializar el fichero PDF
pp = PdfPages('./figures/multipage.pdf')

# SAlvar la figura al PDF
pp.savefig()

# Cerrar el fichero
pp.close()

# Show the plot
plt.show()



# Salvar Figura con Transparencia
#plt.savefig("./figures/demo-figure.png", transparent=True)




"""
Funciones sobre los ejes
ax.arrow()	Arrow
ax.quiver()	2D field of arrows
ax.streamplot()	2D vector fields
ax.hist()	Histogram
ax.boxplot()	Boxplot
ax.violinplot()	Violinplot
ax.pcolor()	Pseudocolor plot
ax.pcolormesh()	Pseudocolor plot
ax.contour()	Contour plot
ax.contourf()	Filled contour plot
ax.clabel()	Labeled contour plot
"""


names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./csv/pima-indians-diabetes.data.csv', names=names)

matplotlib.pyplot.style.use="default"
#Diagrama de caja
data.boxplot()
plt.show()

#Históriograma
data.hist()
plt.show()

#diagrama de dispersión
from pandas.plotting import scatter_matrix
scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()
