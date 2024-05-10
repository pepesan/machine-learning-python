# -*- coding: utf-8 -*-
from bokeh.plotting import figure, output_file, show

# Datos de los puntos
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

# Salida a fichero html
#output_file("lines.html")

# Creado nuevo plot con título y etiquetas de ejes
p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')

# Añade una línea entre los puntos fijados
p.line(x, y, legend_label="Temp.", line_width=2)

# Muestra los resultados
show(p)


