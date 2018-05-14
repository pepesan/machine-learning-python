from bokeh.plotting import figure, output_file, show

# Datos de los puntos
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

# Salida a fichero html
#output_file("lines.html")

# Creado nuevo plot con título y etiquetas de ejes
p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')

# Añade una línea entre los puntos fijados
p.line(x, y, legend="Temp.", line_width=2)

# Muestra los resultados
show(p)


from bokeh.models import ColumnDataSource

#Genera los datos
data = {'x_values': [1, 2, 3, 4, 5],
        'y_values': [6, 7, 2, 3, 6]}

#Crea la fuente de datos
source = ColumnDataSource(data=data)

#Crea la figura
p = figure()
#Crea un círculo
p.circle(x='x_values', y='y_values', source=source)
#Muestra la gráfica
show(p)


# Prepara los datos
x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
y0 = [i**2 for i in x]
y1 = [10**i for i in x]
y2 = [10**(i**2) for i in x]

# Fichero de salida
#output_file("log_lines.html")

# Crea la figura
p = figure(
   tools="pan,box_zoom,reset,save",
   y_axis_type="log", y_range=[0.001, 10**11], title="log axis example",
   x_axis_label='sections', y_axis_label='particles'
)

# añade los renders
p.line(x, x, legend="y=x")
p.circle(x, x, legend="y=x", fill_color="white", size=8)
p.line(x, y0, legend="y=x^2", line_width=3)
p.line(x, y1, legend="y=10^x", line_color="red")
p.circle(x, y1, legend="y=10^x", fill_color="red", line_color="red", size=6)
p.line(x, y2, legend="y=10^x^2", line_color="orange", line_dash="4 4")

# muestra los resultados
show(p)



import numpy as np


# prepare some data
N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = [
    "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)
]

# output to static HTML file (with CDN resources)
output_file("./figures/color_scatter.html", title="color_scatter.py example", mode="cdn")

TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"

# create a new plot with the tools above, and explicit ranges
p = figure(tools=TOOLS, x_range=(0,100), y_range=(0,100))

# add a circle renderer with vectorized colors and sizes
p.circle(x,y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)

# show the results
show(p)