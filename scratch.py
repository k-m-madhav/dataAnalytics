import pandas as pd

# Load the CSV file
df = pd.read_csv('002_product_data.csv', encoding='latin1')
df = pd.read_csv('002_product_data.csv', names=['SKU', 'Description', 'Product Group'], header=None, encoding='latin1')

df2 = pd.read_csv('003_pick_data.csv', encoding='latin1', low_memory=False)
df2 = pd.read_csv('003_pick_data.csv', names=['SKU', 'Warehouse Section', 'Origin', 'Order No', 'Position in Order', 'Pick Volume', 'Unit', 'Date'], header=None, encoding='latin1', low_memory=False)

#columns_to_clean_df2 = ['SKU', 'Order No', 'Position in Order']
#df2[columns_to_clean_df2] = df2[columns_to_clean_df2].apply(lambda x: x.str.replace("'", "", regex=False))

# Display the first few rows of the DataFrame
#print(df.head())
#print(df2.describe(include='all'))

#print(df2.describe())

filteredDf = df2.loc[(df2['Order No'] == '07055448') & (df2['SKU'] == '000004')]
print(filteredDf)
print(filteredDf['Pick Volume'].sum())

#print(df2.head())

#sortedDf = filteredDf.sort_values(by=['Date'])
#print(sortedDf)
#print(sortedDf[['SKU', 'Position in Order']])