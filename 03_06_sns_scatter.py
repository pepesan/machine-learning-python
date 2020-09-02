import seaborn as sns
sns.set(style="ticks")

df = sns.load_dataset("iris")
print(df)
print(df.shape)
plot = sns.pairplot(df, hue="species")

import matplotlib.pyplot as plt
plt.show()