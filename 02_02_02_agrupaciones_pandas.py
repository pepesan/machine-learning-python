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
print(one.sum())
print('groupby both')
letterone = df.groupby(['letter', 'one']).sum()
print(letterone)

print(letterone.index)

letterone = df.groupby(['letter', 'one'], as_index=False).sum()
print(letterone)
