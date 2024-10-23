import pandas as pd

# Load the CSV file
df = pd.read_csv('002_product_data.csv', encoding='latin1')
df = pd.read_csv('002_product_data.csv', names=['SKU', 'Description', 'Product Group'], header=None, encoding='latin1')

df2 = pd.read_csv('003_pick_data.csv', encoding='latin1', low_memory=False)
df2 = pd.read_csv('003_pick_data.csv', names=['SKU', 'Warehouse Section', 'Origin', 'Order No', 'Position in Order', 'Pick Volume', 'Unit', 'Date'], header=None, encoding='latin1', low_memory=False)

# Display the first few rows of the DataFrame
#print(df.head())
#print(df2.describe(include='all'))

#print(df2.describe())
#print(df2.head())

# Sorting the Pick Data based on the Order No and further sorting the rows 
# based on the time stamps. Trying to arrange the data a bit better
sorted_df2 = df2.sort_values(by=['Order No', 'Date'])
print(sorted_df2.head())