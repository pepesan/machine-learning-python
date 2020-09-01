# -*- coding: utf-8 -*-
# Import libraries
import pandas as pd
import sys


print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)


# Our small data set
d = {'one':[1,1,1,1,1],
     'two':[2,2,2,2,2],
     'letter':['a','a','b','b','c']}

# Create dataframe
df = pd.DataFrame(d)
print(df)


# Create group object
one = df.groupby('letter')
print(one)
# Apply sum function
print(one.sum())



letterone = df.groupby(['letter','one']).sum()
print(letterone)


print(letterone.index)


letterone = df.groupby(['letter','one'], as_index=False).sum()
print(letterone)
