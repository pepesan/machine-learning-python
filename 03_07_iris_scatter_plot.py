from sklearn.datasets import load_iris
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource, NumeralTickFormatter
# Cargar datos de Iris
iris = load_iris()

# Separar características y etiquetas
X = iris.data
y = iris.target
# Crear ColumnDataSource
datos = {'sepal_length': X[:, 0], 'sepal_width': X[:, 1], 'petal_length': X[:, 2],
        'petal_width': X[:, 3], 'clase': iris.target_names[y]}
source = ColumnDataSource(data=datos)
# Mapeo de colores por clase
colores = ["red", "green", "blue"]  # Example color list
mapper = factor_cmap('clase', palette=colores, factors=iris.target_names)
# Crear figura de Bokeh
p = figure(
    title="Diagrama de dispersión - Iris",
    x_axis_label="Longitud del sépalo",
    y_axis_label="Ancho del sépalo",
    tools="pan,box_zoom,wheel_zoom,reset",
)

# Diagrama de dispersión
p.circle(
    x='sepal_length',
    y='sepal_width',
    size=10,
    source=source,
    fill_color=mapper,
    fill_alpha=0.6,
    line_color='gray',
)
# Personalizar ejes
p.xaxis.major_label_orientation = 1
p.yaxis.formatter = NumeralTickFormatter(format="0.0")
# Mostrar diagrama
show(p)
