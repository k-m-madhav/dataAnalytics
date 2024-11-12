import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the CSV files
df = pd.read_csv('002_product_data.csv', names=['SKU', 'Description', 'Product Group'], header=None, encoding='latin1')
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

print(sorted_df2.info()) # Gives info about the datatype of the variables (columns)

# Gives info on how many unique values are present for each column
# ex: Warehouse Section = 5, Origin = 2 and so on
print(sorted_df2.nunique())

print(sorted_df2['SKU'].value_counts()) # To see unique items of a column and their counts

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

## ------ Cleaning -ve Pick Volume ------ ##
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

# Create a DataFrame from the combined rows
# combined_df = pd.DataFrame(combined_rows)

negative_indices_removed = list(set(negative_indices) ^ set(missed_indices)) # Removing the common elements between two lists (85 indices)
sorted_df2 = sorted_df2.drop(index=negative_indices_removed).reset_index(drop=True) # Dropping all indices from the dataframe that matched the "Condition"
print(sorted_df2[sorted_df2['Pick Volume'] < 0].info()) # 85 rows less than what we started with

# print(missed_indices) count = 15
# print(sorted_df2.loc[31355201])
# print(sorted_df2[sorted_df2['Order No'] == '03014512'])

## ------ End of 'Cleaning -ve Pick Volume' ------ ##


## ------ To get the Number of Picks per Year 
# by Warehouse Section (Approach 1) [Not the best] ------ ##
sorted_df2['Date'] = pd.to_datetime(sorted_df2['Date'])
print(sorted_df2.info())

# Count picks per order number and warehouse section
pick_counts = sorted_df2.groupby(['Order No', 'Warehouse Section']).size().reset_index(name='pick_count')
print(pick_counts.head())
print(sorted_df2[sorted_df2['Order No'] == '01000002'])

# Now aggregate the start and end timestamps
timestamp_agg = sorted_df2.groupby(['Order No', 'Warehouse Section']).agg(
    start_timestamp=('Date', 'min'),
    end_timestamp=('Date', 'max')
).reset_index()
print(timestamp_agg.head())

# Merge the two DataFrames to get pick counts with timestamps
aggregated_df = pd.merge(timestamp_agg, pick_counts, on=['Order No', 'Warehouse Section'])
print(aggregated_df.head())

# Extract the start year
aggregated_df['start_year'] = aggregated_df['start_timestamp'].dt.year

# Aggregate the counts by year and warehouse section
yearly_counts = aggregated_df.groupby(['start_year', 'Warehouse Section']).agg(
    order_count=('pick_count', 'sum')  # Sum the pick counts for each year and warehouse section
).reset_index()
print(yearly_counts.head())

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=yearly_counts, x='start_year', y='order_count', hue='Warehouse Section')
plt.title('Number of Picks per Year by Warehouse Section')
plt.xlabel('Year')
plt.ylabel('Number of Picks')
plt.legend(title='Warehouse Section')
plt.show()

## ------ End of 'Number of Picks per Year by Warehouse Section (Approach 1)' ------ ##


## ------ Monthly and Quarterly Analysis of the No. of Picks per WH Section (Approach 2) ------ ##
sorted_df2['Year'] = sorted_df2['Date'].dt.year
sorted_df2['Month'] = sorted_df2['Date'].dt.month # For monthly aggregation
sorted_df2['Quarter'] = sorted_df2['Date'].dt.to_period('Q') # For quarterly aggregation
sorted_df2.info()

monthly_picks = sorted_df2.groupby(['Year', 'Month', 'Warehouse Section']).size().reset_index(name='Picks')
print(monthly_picks.head())

quarterly_picks = sorted_df2.groupby(['Year', 'Quarter', 'Warehouse Section']).size().reset_index(name='Picks')
print(quarterly_picks.head())

#monthly_pivot = monthly_picks.pivot_table(index=['Year', 'Month'], columns='Warehouse Section', values='Picks', fill_value=0)
#quarterly_pivot = quarterly_picks.pivot_table(index=['Year', 'Quarter'], columns='Warehouse Section', values='Picks', fill_value=0)

## ------ End of 'Approach 2' ------ ##


##  ------ Fix the Repeating Order Numbers by creating Unique Order Numbers 
# based on the assumption that an order is assumed to be completed within 5 days ------ ##

#sorted_df2['Date'] = pd.to_datetime(sorted_df2['Date']) already done somewhere above so commenting
pick_data = sorted_df2.copy(deep=True) # creating a separate copy of sorted_df2 DF. This new DF will have the 'Unique Order No' column

#pick_data = pick_data.sort_values(by=['Order No', 'Date']).reset_index(drop=True) This sort needs to be done for faster calc. It's already done above so commenting out
pick_data['Time Difference'] = pick_data.groupby('Order No')['Date'].diff().dt.days # Calculate the time difference between consecutive rows within each Order Number group (Took 37m 5.6s to execute)

# Fastest approach acc to chat-gpt (but didn't test it out)
# # Calculate time difference directly using shift on sorted data
# pick_data['Time Difference'] = (pick_data['Date'] - pick_data['Date'].shift()).dt.days
# # Reset differences to NaN where Order No changes
# pick_data['Time Difference'] = pick_data['Time Difference'].where(
#     pick_data['Order No'] == pick_data['Order No'].shift()
# )

#print(pick_data[pick_data['Order No'] == '01000002'])
time_diffs = pick_data['Time Difference'].dropna() # Drop NaN values (first occurrence in each group)
print(time_diffs.value_counts().sort_index())
time_diffs_non_0_days = time_diffs[time_diffs != 0.0] # filters out all entries where the value is 0.0
freq_non_0_days = time_diffs_non_0_days.value_counts().sort_index()

# Plotting a distribution of time differences excl 0
plt.figure(figsize=(10, 6))
plt.hist(time_diffs_non_0_days, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Days Between Orders')
plt.ylabel('Frequency')
plt.title('Distribution of Time Differences Between Orders (Excl 0)')
plt.show()

threshold_days = 5 # Set the threshold in days (to differentiate legitimate order repeats from duplicates) (Chat-GPT recommends to assume that an order takes 5 days to complete)
pick_data['New Version Flag'] = (pick_data['Time Difference'] > threshold_days).fillna(False) # Create a flag where time difference is greater than threshold (indicating a new version)
#print(pick_data[pick_data['Order No'] == '01000002'])
pick_data['Version'] = pick_data.groupby('Order No')['New Version Flag'].cumsum() + 1 # Calculate the cumulative sum of the new version flag to create a version counter within each Order Number
pick_data['Unique Order No'] = pick_data['Order No'].astype(str) + "_v" + pick_data['Version'].astype(str) # Creating the Unique Order No column using the Order Number and Version 
pick_data = pick_data.drop(columns=['Time Difference', 'New Version Flag', 'Version']) # Dropping the temporary columns
#print(pick_data[pick_data['Order No'] == '01000002'])
pick_data = pick_data.sort_values(by=['Unique Order No', 'Date']).reset_index(drop=True) # Sort by 'Unique Order No' and 'Date'
#print(pick_data.nunique())

## ------ End of 'Fix the Repeating Order No by creating Unique Order No' ------ ##


## ----- Merging the Datasets ----- ##
merged_df = pd.merge(pick_data, df, on='SKU', how='left') # A left join includes all rows from the left DataFrame (pick_data), and matched rows from the right DataFrame (df). If there's no match, NaN values are filled in for columns from the second datframe, df.
print(merged_df.isnull().sum())
null_rows_product_group = merged_df[merged_df['Product Group'].isnull()] # Show rows with Product Group being null
print(null_rows_product_group)

## ----- End of 'Merging the Datasets' ----- ##


## ----- Creating a new DF called 'unique_order_details' ----- ##

# Grouping by 'Unique Order No' and getting the first and last timestamp for each order
unique_order_details = pick_data.groupby('Unique Order No').agg(
    Start_Time=('Date', 'first'),  # First timestamp for each order
    End_Time=('Date', 'last')      # Last timestamp for each order
).reset_index()

## ++ Enter your code here! ++ ##


## ++ Enter your code here! ++ ##

## ----- End of 'unique_order_details DF' ----- ##


## ------ EDA Univariate Analysis ------ ## 

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