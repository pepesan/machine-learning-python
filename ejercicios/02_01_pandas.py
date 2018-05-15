# -*- coding: utf-8 -*-
#Step 1. Import the necessary libraries
import pandas as pd
import numpy as np

#Step 2. Import the dataset from this address.
#Step 3. Assign it to a variable called chipo.

#url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
url = './chipotle.tsv'
chipo = pd.read_csv(url, sep='\t')

#Step 4. See the first 10 entries
print("#4")
print(chipo.head(10))
# chipo['choice_description'][4]

#Step 5. What is the number of observations in the dataset?
print("#5")
print(chipo.info())#

# OR

print(chipo.shape[0])
# 4622 observations

#Step 6. What is the number of columns in the dataset?
print("#6")
print(chipo.shape[1])

#Step 7. Print the name of all the columns.
print("#7")
print(chipo.columns)

#Step 8. How is the dataset indexed?
print("#8")
print(chipo.index)

#Step 9. Which was the most ordered item?
print("#9")
c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
print(c.head(1))
#Step 10. How many items were ordered?
print("#10")
c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
#print(c.head(10))
print(c)
#Step 11. What was the most ordered item in the choice_description column?
c = chipo.groupby('choice_description').sum()
c = c.sort_values(['quantity'], ascending=False)
print("#11")
print(c.head(1))
# Diet Coke 159

#Step 12. How many items were orderd in total?
total_items_orders = chipo.quantity.sum()
print("#12")
print(total_items_orders)

#Step 13. Turn the item price into a float
dollarizer = lambda x: float(x[1:-1])
chipo.item_price = chipo.item_price.apply(dollarizer)
print("#13")
print(chipo.item_price[0])
#Step 14. How much was the revenue for the period in the dataset?
revenue = (chipo['quantity']* chipo['item_price']).sum()
print("#14")
print('Revenue was: $' + str(np.round(revenue,2)))


#Step 15. How many orders were made in the period?
print("#15")
print(chipo.order_id.value_counts().count())

#Step 16. What is the average amount per order?
print("#16")
chipo['revenue'] = chipo['quantity']* chipo['item_price']
order_grouped = chipo.groupby(by=['order_id']).sum()
print(order_grouped.mean()['revenue'])


# Or

#chipo.groupby(by=['order_id']).sum().mean()['item_price']

#Step 17. How many different items are sold?
print("#17")
chipo.item_name.value_counts().count()