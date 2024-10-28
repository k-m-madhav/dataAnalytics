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