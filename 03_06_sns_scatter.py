import seaborn as sns
sns.set(style="ticks")

df = sns.load_dataset("iris")
plot=sns.pairplot(df, hue="species")

import matplotlib.pyplot as plt
plt.show()