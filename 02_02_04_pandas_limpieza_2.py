import pandas as pd
import numpy as np
# ref https://github.com/realpython/python-data-cleaning/blob/master/Datasets/BL-Flickr-Images-Book.csv
# ref blog: https://realpython.com/python-data-cleaning-numpy-pandas/
df = pd.read_csv("csv/BL-Flickr-Images-Book.csv")
print(df)

print(df.head())
print(df.info())
to_drop = ['Edition Statement',
'Corporate Author',
'Corporate Contributors',
'Former owner',
'Engraver',
'Contributors',
'Issuance type',
'Shelfmarks']

df.drop(to_drop, inplace=True, axis=1)
print("Dropped")
print(df)

# índice unico
df['Identifier'].is_unique
# usado como index del DF
# devuelve un df copiado
#df = df.set_index('Identifier')
# devuelve sobre el mismo df
df.set_index('Identifier', inplace=True)

# convertir a números una columna
extr = df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
df['Date of Publication'] = pd.to_numeric(extr)
print(df['Date of Publication'].dtype)


# limpiar datos
pub = df['Place of Publication']
# mascara de si contiene londres
london = pub.str.contains('London')
oxford = pub.str.contains('Oxford')
df['Place of Publication'] = np.where(london, 'London',
                                      np.where(oxford, 'Oxford',
                                               pub.str.replace('-', ' ')))

print(df['Place of Publication'].head())

print(df)



## Maś limpieza
university_towns = []
with open('Datasets/university_towns.txt') as file:
    for line in file:
        if '[edit]' in line:
            # Remember this `state` until the next is found
            state = line
        else:
            # Otherwise, we have a city; keep `state` as last-seen
            university_towns.append((state, line))

towns_df = pd.DataFrame(
    university_towns,
    columns=['State', 'RegionName'])


def get_citystate(item):
    if ' (' in item:
        return item[:item.find(' (')]
    elif '[' in item:
        return item[:item.find('[')]
    else:
        return item

towns_df =  towns_df.map(get_citystate)

