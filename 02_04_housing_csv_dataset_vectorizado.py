# Python code to illustrate
# regression using data set
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import numpy as np

# Load CSV and columns
df = pd.read_csv("csv/Housing.csv")


# Vectoriza el campo "driveway"
df["driveway_yes"] = (df["driveway"] == "yes").astype("int")
df["driveway_no"] = (df["driveway"] == "no").astype("int")

# Guarda el DataFrame actualizado
df.to_csv("data_vectorized.csv")

print(df.head())
