# -*- coding: utf-8 -*-
# Import libraries
import pandas as pd
import sys

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)

# Our small data set
d = {'one': [1, 1, 4, 4, 5],
     'two': [2, 2, 3, 3, 5],
     'letter': ['a', 'a', 'b', 'b', 'c']}

# Create dataframe
df = pd.DataFrame(d)
print(df)

# Create group object
print('groupby')
one = df.groupby('letter')
print(one)
# Apply sum function
print("sum")
print(one.sum())
# Apply count function
print("count")
print(one.count())
print('groupby both letter and one result sum')
letterone = df.groupby(['letter', 'one']).sum()
print(letterone)

print(letterone.index)

letterone = df.groupby(['letter', 'one'], as_index=False).sum()
print(letterone)


df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10], 'C': ['a', 'b', 'c', 'a', 'b']})

# Agrupar por 'C' y calcular la suma de 'A' y 'B'
print(df.groupby('C').agg(sum_A=('A', 'sum'), sum_B=('B', 'sum')))
