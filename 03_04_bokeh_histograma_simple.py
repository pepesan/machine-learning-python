import pandas as pd
from bokeh.plotting import figure, output_file, show

# Cargamos los datos (assuming "datos.csv" exists)
datos = pd.read_csv("csv/histograma.csv")

# Seleccionamos la columna que queremos representar
columna = datos["Columna"]

# Definimos el rango de valores para el histograma
rango_min = min(columna)
rango_max = max(columna)

# Creamos el histograma (using either 'width' or 'outer_width')
histograma = figure(title="Histograma de " + columna.name,
                    x_axis_label=columna.name,
                    y_axis_label="Frecuencia",
                    # Use 'width' if your Bokeh version supports it
                    width=500,  # Adjust width as desired
                    # Otherwise, use 'outer_width' for older versions
                    # outer_width=500
                    )

# Añadimos los datos al histograma
histograma.quad(top=columna.value_counts(),
                bottom=0,
                left=rango_min,
                right=rango_max,
                fill_color="navy",
                line_color="white")

# Mostramos el histograma
output_file("histograma.html")
show(histograma)

#
# # Creamos el histograma
# histograma = figure(title="Histograma de " + columna.name,
#                     x_axis_label=columna.name,
#                     y_axis_label="Frecuencia",
#                     plot_width=500,
#                     plot_height=300)
#
# # Añadimos los datos al histograma
# histograma.quad(top=columna.value_counts(),
#                 bottom=0,
#                 left=rango_min,
#                 right=rango_max,
#                 fill_color="navy",
#                 line_color="white")
#
# # Mostramos el histograma
# output_file("histograma.html")
# show(histograma)
