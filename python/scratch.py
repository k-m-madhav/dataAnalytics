import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the CSV files
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

# Converting the sorted_df2 DF to a CSV file
# sorted_df2.to_csv("003_pick_data_sorted.csv", index=False)

# To print all rows corresponding to the specified SKU
# from product_data.csv
# skus = ['A80704', '387126', '283136']
# print(df[df['SKU'].isin(skus)])

# Gives info about the datatype of the variables (columns)
print(sorted_df2.info())

# Gives info on how many unique values are present for each column
# ex: Warehouse Section = 5, Origin = 2 and so on
print(sorted_df2.nunique())

# sorted_df2.isnull().sum() is used to get the number of missing records in each column
print(sorted_df2.isnull().sum())
# There are no null values!

# Provide a statistics summary of data belonging 
# to numerical datatype such as int, float
print(sorted_df2.describe().T)

# 100 rows which have -ve Pick Vol.
print(sorted_df2[sorted_df2['Pick Volume'] < 0])
print(sorted_df2[sorted_df2['Pick Volume'] < 0].nunique())

# "Not really needed. Was doing some analysis to understand 
# how many Order No.s have multiple -ve Pick Vol"
# negative_pick_volume_df = sorted_df2[sorted_df2['Pick Volume'] < 0]
# order_no_counts = negative_pick_volume_df['Order No'].value_counts()
# non_unique_order_no = order_no_counts[order_no_counts > 1]
# print(non_unique_order_no)

## Cleaning -ve Pick Volume ##
pick_volume = 'Pick Volume'
negative_indices = sorted_df2.index[sorted_df2[pick_volume] < 0].to_list() # storing all indices that have -ve Pick Vol in a list
print(len(negative_indices))

combined_rows = [] # not required
rows_to_combine = [] # not required
missed_indices = [] # All the indices which didn't match the "Condition" are stored in this list
for index in negative_indices:
    if index > 0: # a useful validation but doesn't affect our case
       row_above = sorted_df2.loc[index - 1] # The row which is literally above the one that has -ve pick vol
       negative_row = sorted_df2.loc[index] # The row with -ve pick vol

       if(row_above.drop(labels=pick_volume).equals(negative_row.drop(labels=pick_volume))): # "Condition" to validate if all other variables (column values) except Pick Vol are an exact match
        rows_to_combine.append((row_above, negative_row)) # not really required, created it for tracking
        combined_row = row_above.copy()
        combined_row[pick_volume] = row_above[pick_volume] + negative_row[pick_volume]
        sorted_df2.loc[index - 1] = combined_row # replacing "row_above" with "combined_row"
        combined_rows.append(combined_row) # again, created for tracking purpose
       else:
          missed_indices.append(index)

negative_indices_removed = list(set(negative_indices) ^ set(missed_indices)) # Removing the common elements between two lists (85 indices)
sorted_df2 = sorted_df2.drop(index=negative_indices_removed).reset_index(drop=True) # Dropping all indices from the dataframe that matched the "Condition"
print(sorted_df2[sorted_df2['Pick Volume'] < 0].info()) # 85 rows less than what we started with

# print(missed_indices) count = 15
# print(sorted_df2.loc[31355201])
# print(sorted_df2[sorted_df2['Order No'] == '03014512'])

## EDA Univariate Analysis ## 
# Analyzing/visualizing the dataset by taking one variable at a time

# Separating Numerical and Categorical variables for easy analysis
cat_cols = sorted_df2.select_dtypes(include=['object']).columns
num_cols = sorted_df2.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)

# Doing a Univariate Analysis using Histogram and Box Plot for continuous variables
for col in num_cols:
    print(col)
    print('Skew :', round(sorted_df2[col].skew(), 2))
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    sorted_df2[col].hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=sorted_df2[col])
    plt.show()

# Categorical Variables are being visualized using a count plot
fig, axes = plt.subplots(1, 2, figsize = (18, 18))
fig.suptitle('Bar plot for all categorical variables in the dataset')
sns.countplot(ax=axes[0], x = 'Warehouse Section', data = sorted_df2, color = 'blue', order = sorted_df2['Warehouse Section'].value_counts().index)
sns.countplot(ax=axes[1], x = 'Unit', data = sorted_df2, color = 'blue', order = sorted_df2['Unit'].value_counts().index)
# axes[0].tick_params(labelrotation=45)
# axes[1].tick_params(labelrotation=90)

# Function for log transformation of the column
# def log_transform(data,col):
#     for colname in col:
#         if (data[colname] == 1.0).all():
#             data[colname + '_log'] = np.log(data[colname]+1)
#         else:
#             data[colname + '_log'] = np.log(data[colname])
#     data.info()

# log_transform(sorted_df2,['Position in Order','Pick Volume'])


# extra comment

# A brand new commit 