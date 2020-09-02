#importación de los modelos de datos de sklearn
from sklearn import datasets
cancer = datasets.load_breast_cancer(as_frame=True)
print(cancer.DESCR)
print(cancer.feature_names)
print(cancer.target_names)
print(cancer.target.shape)
print(cancer.data.shape)
#print(cancer.frame)
print(cancer.frame.head(1))
