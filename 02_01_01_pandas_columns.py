# -*- coding: utf-8 -*-

#importacion de pandas
import pandas as pd
import sys

print('Python version ' + sys.version)
print('Pandas version: ' + pd.__version__)


# Our small data set
d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Create dataframe
df = pd.DataFrame(d)
print(df)


# Lets change the name of the column
df.columns = ['Rev']
print(df)


# Lets add a column
df['NewCol'] = 5
print(df)


# Lets modify our new column
df['NewCol'] = df['NewCol'] + 1
print(df)


# We can delete columns
del df['NewCol']
print(df)


# Lets add a couple of columns
df['test'] = 3
df['col'] = df['Rev']
print(df)


# If we wanted, we could change the name of the index
i = ['a','b','c','d','e','f','g','h','i','j']
df.index = i
print(df)


print(df.loc['a'])



# df.loc[inclusive:inclusive]
print(df.loc['a':'d'])


# df.iloc[inclusive:exclusive]
# Note: .iloc is strictly integer position based. It is available from [version 0.11.0] (http://pandas.pydata.org/pandas-docs/stable/whatsnew.html#v0-11-0-april-22-2013)
print(df.iloc[0:3])


print(df['Rev'])


print(df[['Rev', 'test']])


# df.ix[rows,columns]
# replaces the deprecated ix function
#df.ix[0:3,'Rev']
print(df.loc[df.index[0:3],'Rev'])


# replaces the deprecated ix function
#df.ix[5:,'col']
print(df.loc[df.index[5:],'col'])



# replaces the deprecated ix function
#df.ix[:3,['col', 'test']]
print(df.loc[df.index[:3],['col', 'test']])


# Select top N number of records (default = 5)
print(df.head())


# Select bottom N number of records (default = 5)
print(df.tail())
