# !pip install seaborn kagglehub
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos de ejemplo
tips = sns.load_dataset("tips")

# Crear un relplot
sns.relplot(data=tips, x="total_bill", y="tip", hue="sex", style="smoker", size="size")
plt.show()
# interesante porque muestra reaciones entre variables dentro de un scatter plotç
sns.relplot(data=tips, x="total_bill", y="tip", kind="line")
plt.show()

# Diferencia entre relplot() y scatterplot()/lineplot()
# relplot() es una función de nivel alto que genera una figura y usa FacetGrid internamente, lo que permite dividir los datos en múltiples subgráficos.
# sns.scatterplot() y sns.lineplot() son funciones de nivel bajo, que crean gráficos individuales sin la capacidad de faceteo automático.


import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos de ejemplo
tips = sns.load_dataset("tips")

# Crear un catplot
sns.catplot(data=tips, x="day", y="total_bill", hue="sex", kind="box")
plt.show()
# Gráfico de dispersión categórico (strip plot, por defecto)
sns.catplot(data=tips, x="day", y="total_bill", kind="strip")
plt.show()
# Gráfico de enjambre (swarm plot)
sns.catplot(data=tips, x="day", y="total_bill", kind="swarm")
plt.show()
# Gráfico de caja sin valores atípicos (boxen plot)
sns.catplot(data=tips, x="day", y="total_bill", kind="boxen")
plt.show()
# Gráfico de barras (bar plot)
sns.catplot(data=tips, x="day", y="total_bill", kind="bar")
plt.show()
# Gráfico de conteo (count plot)
sns.catplot(data=tips, x="day", kind="count")
plt.show()
# Gráfico de violín (violin plot)
sns.catplot(data=tips, x="day", y="total_bill", kind="violin")
plt.show()
# Gráfico de puntos (point plot)
sns.catplot(data=tips, x="day", y="total_bill", kind="point")
plt.show()

# Diferencias entre catplot() y boxplot()/barplot()
# catplot() es una función de alto nivel que usa FacetGrid, lo que permite crear múltiples subgráficos automáticamente.
# Funciones como sns.boxplot() o sns.barplot() son de nivel bajo, creando gráficos individuales sin faceteo automático.

import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos de ejemplo
tips = sns.load_dataset("tips")

# Crear un displot
sns.displot(data=tips, x="total_bill", kind="hist", bins=20)
plt.show()

# Crear un Histograma con estimación de densidad (KDE)
sns.displot(data=tips, x="total_bill", kde=True)
plt.show()

# Crear un Gráfico de densidad (KDE plot)
sns.displot(data=tips, x="total_bill", kind="kde")
plt.show()

# Crear un Gráfico de dispersión (ECDF)
sns.displot(data=tips, x="total_bill", kind="ecdf")
plt.show()
