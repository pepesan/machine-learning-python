import pandas as pd

df = pd.DataFrame([[1, 2.12], [3.356, 4.567], [4.356, 5.567]])
print(df.shape)
print(df)
df = df.applymap(lambda x: x-1)
print(df.shape)
print(df)
