# -*- coding: utf-8 -*-
import sklearn
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

mglearn.plots.plot_pca_illustration()
cancer = load_breast_cancer()
print(cancer.target_names)

pca = PCA(n_components=2)
pca.fit(cancer.data)

transformada = pca.transform(cancer.data)

print(cancer.data.shape)

print(transformada.data.shape)


mglearn.discrete_scatter(transformada[:, 0], transformada[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.xlabel("PC1")
plt.ylabel("PC2")
# enseña los datos y su procesado
plt.show()


escala = MinMaxScaler()
escala.fit(cancer.data)
escalada = escala.transform(cancer.data)
pca.fit(escalada)
transformada = pca.transform(escalada)
mglearn.discrete_scatter(transformada[:, 0], transformada[:, 1], cancer.target)
plt.legend(cancer.target_names, loc="best")
plt.gca()
plt.xlabel("PC1")
plt.ylabel("PC2")
# enseña los datos y su procesado
plt.show()
