import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the CSV files
df = pd.read_csv('002_product_data.csv', names=['SKU', 'Description', 'Product_Group'], header=None, encoding='latin1', engine='python', error_bad_lines=False, quoting=3)
df2 = pd.read_csv('003_pick_data.csv', names=['SKU', 'Warehouse_Section', 'Origin', 'Order_No', 'Position_in_Order', 'Pick_Volume', 'Unit', 'Date'], header=None, encoding='latin1', low_memory=False)

## ----- Handling Mixed Datatypes ----- ##
# print(df2['Warehouse_Section'].apply(type).value_counts()) O/P: <class 'str'>    33888990
df2[['Warehouse_Section', 'Unit']] = df2[['Warehouse_Section', 'Unit']].astype('string')  #pandas' StringDtype ('string') is recommended for better memory efficiency and optimized string operations
df2['Date'] = pd.to_datetime(df2['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df2[['SKU', 'Order_No']] = df2[['SKU', 'Order_No']].astype('string')
## ----- End of 'Handling Mixed Datatypes' ----- ##

## ----- SKU and Unit inconsistency ----- ##
# Group by 'SKU' and get unique 'Unit' values
unit_inconsistencies = df2.groupby('SKU')['Unit'].unique()

# Filter SKUs that have more than one unique 'Unit', sort the units and then create the summary with both unique units and their counts
inconsistent_skus_summary = pd.DataFrame({
    'Unique_Units': unit_inconsistencies[unit_inconsistencies.apply(lambda x: len(x) > 1)]
                            .apply(lambda x: sorted(list(x))),  # Sort units to avoid discrepancies such as [Mt,St] and [St,Mt] being considered as two unique combinations
    'Count_of_Unique_Units': unit_inconsistencies[unit_inconsistencies.apply(lambda x: len(x) > 1)]
                            .apply(lambda x: len(x))  # Count the unique units # All values are 2
})
print(inconsistent_skus_summary['Unique Units'].value_counts()) # other unit combinations seem fine except [Mt,St] and it has a freq of 7

## ----- End of 'SKU and Unit inconsistency' ----- ##

## ----- Cleaning Product Data dataset ----- ##
print(df.isnull().sum()) # didn't find any null values because it considered "" i.e. Empty String, in the Description as a valid entry
print(df.nunique()) # SKU = 2199644, Description = 1422864, Product_Group = 18
df['Description'] = df['Description'].replace('""', np.nan) # Replace exact `""` with NaN
df['Description'] = df['Description'].str.strip('"') # Strip any leading or trailing double quotes from Description

descriptions_with_multiple_skus = df.groupby('Description')['SKU'].nunique()
descriptions_with_multiple_skus = descriptions_with_multiple_skus[descriptions_with_multiple_skus > 1]

## ----- End of 'Cleaning Product Data dataset' ----- ## 


## ----- Removing duplicate picks from Pick Data dataset ----- ##
print(df2.duplicated().sum())
duplicate_picks = df2[df2.duplicated(keep=False)]
df2_transformed = df2.drop_duplicates()

## ----- End of 'Removing duplicate picks from Pick Data dataset' ----- ##


##  ------ Fix the Repeating Order Numbers by creating Unique Order Numbers 
# based on the assumption that an order is assumed to be completed within 5 days ------ ##

df2_transformed = df2_transformed.sort_values(by=['Order_No', 'Date']).reset_index(drop=True) # Sort by 'Unique_Order_No' and 'Date'

# df2_transformed['Time_Difference'] = df2_transformed.groupby('Order_No')['Date'].diff().dt.days # Calculate the time difference between consecutive rows within each Order Number group (Took 37m 5.6s to execute)

df2_transformed['Time_Difference'] = (df2_transformed['Date'] - df2_transformed['Date'].shift()).dt.days # Fastest approach acc to chat-gpt to calculate time difference. Directly using shift on sorted data
# Reset differences to NaN where Order_No changes
df2_transformed['Time_Difference'] = df2_transformed['Time_Difference'].where(
    df2_transformed['Order_No'] == df2_transformed['Order_No'].shift()
)

#print(df2_transformed[df2_transformed['Order_No'] == '01000002'])
time_diffs = df2_transformed['Time_Difference'].dropna() # Drop NaN values (first occurrence in each group)
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

threshold_days = 5 # Set the threshold in days (to differentiate legitimate order repeats from duplicates, Chat-GPT recommends to assume that an order takes 5 days to complete)
df2_transformed['New_Version_Flag'] = (df2_transformed['Time_Difference'] > threshold_days).fillna(False) # Create a flag where time difference is greater than threshold (indicating a new version)
#print(df2_transformed[df2_transformed['Order_No'] == '01000002'])
df2_transformed['Version'] = df2_transformed.groupby('Order_No')['New_Version_Flag'].cumsum() + 1 # Calculate the cumulative sum of the new version flag to create a version counter within each Order Number
df2_transformed['Unique_Order_No'] = df2_transformed['Order_No'].astype(str) + "_v" + df2_transformed['Version'].astype(str) # Creating the Unique_Order_No column using the Order Number and Version 
df2_transformed = df2_transformed.drop(columns=['Time_Difference', 'New_Version_Flag', 'Version']) # Dropping the temporary columns
#print(df2_transformed[df2_transformed['Order_No'] == '01000002'])
df2_transformed = df2_transformed.sort_values(by=['Unique_Order_No', 'Date']).reset_index(drop=True) # Sort by 'Unique_Order_No' and 'Date'
#print(df2_transformed.nunique())

# Testing if an Edge Case is covered. Start Date is 2015-12-30 and End Date is 2016-01-03 which is within the 5 day threshold and needs to be considered as 1 Unique Order
# df2_transformed['Year'] = df2_transformed['Date'].dt.year
# diff_years = df2_transformed.groupby('Unique_Order_No')['Year'].nunique()
# unique_order_no_with_diff_years = diff_years[diff_years > 1].index
# print("Unique Order numbers with diff years: ", unique_order_no_with_diff_years)
# print(df2_transformed[df2_transformed['Unique_Order_No'] == '02286155_v2'])

## ------ End of 'Fix the Repeating Order_No by creating Unique_Order_No' ------ ##


## ----- 0 Pick_Vol ----- ##

# Step 1: Create a flag for rows with Pick_Vol = 0 and Pick_Vol > 0
df2_transformed['Pick_Vol_Zero_Flag'] = df2_transformed['Pick_Volume'] == 0
df2_transformed['Pick_Vol_Positive_Flag'] = df2_transformed['Pick_Volume'] > 0

# Step 2: Group by Unique_Order_No, SKU, and Position_in_Order
grouped = df2_transformed.groupby(['Unique_Order_No', 'SKU', 'Position_in_Order'])

# Step 3: Check conditions within each group
matching_orders = grouped.agg({
    'Pick_Vol_Zero_Flag': 'any',      # Check if Pick_Vol = 0 exists
    'Pick_Vol_Positive_Flag': 'any'   # Check if Pick_Vol > 0 exists
}).reset_index()

# Step 4: Filter groups that meet the criteria
matching_orders['Has_Zero_and_Positive_Pick_Vol'] = matching_orders['Pick_Vol_Zero_Flag'] & matching_orders['Pick_Vol_Positive_Flag']
matching_orders['Has_Zero_but_No_Positive_Pick_Vol'] = matching_orders['Pick_Vol_Zero_Flag'] & ~matching_orders['Pick_Vol_Positive_Flag']

# Step 5: Get the count of Unique_Order_No that meet the criteria and the count of the ones that don't
orders_with_zero_and_positive = matching_orders[matching_orders['Has_Zero_and_Positive_Pick_Vol']]['Unique_Order_No'].unique() # 145,829
problematic_orders = matching_orders[matching_orders['Has_Zero_but_No_Positive_Pick_Vol']]['Unique_Order_No'].unique() # 2215
overlap_orders = set(orders_with_zero_and_positive).intersection(problematic_orders) # 76 Unique Orders come under both categories

# Analyze the strictly problematic orders
strictly_problematic_orders = list(set(problematic_orders) - overlap_orders)

# Filter all rows for strictly problematic orders
strictly_problematic_orders_df = df2_transformed[df2_transformed['Unique_Order_No'].isin(strictly_problematic_orders)]

# Count unique combinations of Unique_Order_No, SKU and Position_in_Order with Pick_Vol = 0
zero_pick_stats = strictly_problematic_orders_df[strictly_problematic_orders_df['Pick_Volume'] == 0].groupby(
    ['Unique_Order_No', 'SKU', 'Position_in_Order']
).size().reset_index(name='Count')

# Count unique combinations of Warehouse_Section and Origin with Pick_Vol = 0
warehouse_and_origin_pick_stats = strictly_problematic_orders_df[strictly_problematic_orders_df['Pick_Volume'] == 0].groupby(
    ['Warehouse_Section', 'Origin']
).size().reset_index(name='Count')

## ----- End of '0 Pick_Vol' ----- ##

# Display the first few rows of the DataFrame
#print(df.head())
#print(df2.describe(include='all'))

#print(df2.describe())
#print(df2.head())

# To print all rows corresponding to the specified SKU
# from product_data.csv
# skus = ['A80704', '387126', '283136']
# print(df[df['SKU'].isin(skus)])

print(df2_transformed.info()) # Gives info about the datatype of the variables (columns)
print(df2_transformed.nunique()) # Gives info on how many unique values are present for each column. ex: Warehouse_Section = 5, Origin = 2 and so on
print(df2_transformed['SKU'].value_counts()) # To see unique items of a column and their counts

# Is used to get the number of missing records in each column
print(df2_transformed.isnull().sum())
# There are no null values!

print(df2_transformed.describe().T) # Provide a statistics summary of data belonging to numerical datatype such as int, float

# 100 rows which have -ve Pick Vol.
print(df2_transformed[df2_transformed['Pick_Volume'] < 0])
print(df2_transformed[df2_transformed['Pick_Volume'] < 0].nunique())

# "Not really needed. Was doing some analysis to understand 
# how many Order_No.s have multiple -ve Pick Vol"
# negative_pick_volume_df = df2_transformed[df2_transformed['Pick_Volume'] < 0]
# order_no_counts = negative_pick_volume_df['Order_No'].value_counts()
# non_unique_order_no = order_no_counts[order_no_counts > 1]
# print(non_unique_order_no)

## ------ Cleaning -ve Pick_Volume ------ ##
pick_volume = 'Pick_Volume'
negative_indices = df2_transformed.index[df2_transformed[pick_volume] < 0].to_list() # storing all indices that have -ve Pick Vol in a list
print(len(negative_indices))

missed_indices = [] # All the indices which didn't match the "Condition" are stored in this list
missed_indices_sum_0 = [] # All the indices which matched the "Condition" but the sum of Pick_Volume equalled 0
indices_to_drop = [] # List to collect indices that need to be dropped
unique_order_nos_condition_failed = []
unique_order_nos_pick_volume_sum_0 = []
unique_order_nos_dropped = []

columns_to_drop_before_compare = df2_transformed.columns.difference([pick_volume, 'Pick_Vol_Positive_Flag', 'Pick_Vol_Zero_Flag'])

for index in negative_indices:
    if index > 0:  # Ensure there's a row above to compare with
        row_above = df2_transformed.loc[index - 1, columns_to_drop_before_compare]  # Row above the one with negative pick volume
        negative_row = df2_transformed.loc[index, columns_to_drop_before_compare]   # Row with negative pick volume. Selecting only relevant columns for comparison

        # Compare rows excluding 'pick_volume'
        if(row_above.equals(negative_row)):
            combined_pick_volume = df2_transformed.at[index - 1, pick_volume] + df2_transformed.at[index, pick_volume] # Combine the 'pick_volume' values

            if combined_pick_volume > 0:
                df2_transformed.at[index - 1, pick_volume] = combined_pick_volume # Update the pick_volume for the row_above
                indices_to_drop.append(index) # Add the index of the negative pick_volume row to the drop list
                unique_order_nos_dropped.append(df2_transformed.at[index, 'Unique_Order_No'])  # Store the Unique_Order_No for the row being dropped
            else:
                missed_indices_sum_0.append(index) # Track where combined Pick_Volume is <= 0
                unique_order_nos_pick_volume_sum_0.append(df2_transformed.at[index, 'Unique_Order_No'])  # Store the Unique_Order_No for the row where combined Pick_Volume <= 0
        else:
            missed_indices.append(index) # Track rows where they don't match
            unique_order_nos_condition_failed.append(df2_transformed.at[index, 'Unique_Order_No'])  # Store the Unique_Order_No where the 'Condition' failed

df2_transformed = df2_transformed.drop(indices_to_drop) # Drop rows based on indices
df2_transformed = df2_transformed.reset_index(drop=True) # Reset index to create a clean, sequential index
print(df2_transformed.info()) # 71 rows less

# print(missed_indices) # count = 15
# print(df2_transformed.loc[31355201])
# print(df2_transformed[df2_transformed['Order_No'] == '03014512'])

## ------ End of 'Cleaning -ve Pick_Volume' ------ ##


## ------ To get the Number of Picks per Year 
# by Warehouse_Section (Approach 1) [Not the best] ------ ##
print(df2_transformed.info())

# Count picks per order number and warehouse section
pick_counts = df2_transformed.groupby(['Order_No', 'Warehouse_Section']).size().reset_index(name='pick_count')
print(pick_counts.head())
print(df2_transformed[df2_transformed['Order_No'] == '01000002'])

# Now aggregate the start and end timestamps
timestamp_agg = df2_transformed.groupby(['Order_No', 'Warehouse_Section']).agg(
    start_timestamp=('Date', 'min'),
    end_timestamp=('Date', 'max')
).reset_index()
print(timestamp_agg.head())

# Merge the two DataFrames to get pick counts with timestamps
aggregated_df = pd.merge(timestamp_agg, pick_counts, on=['Order_No', 'Warehouse_Section'])
print(aggregated_df.head())

# Extract the start year
aggregated_df['start_year'] = aggregated_df['start_timestamp'].dt.year

# Aggregate the counts by year and warehouse section
yearly_counts = aggregated_df.groupby(['start_year', 'Warehouse_Section']).agg(
    order_count=('pick_count', 'sum')  # Sum the pick counts for each year and warehouse section
).reset_index()
print(yearly_counts.head())

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=yearly_counts, x='start_year', y='order_count', hue='Warehouse_Section')
plt.title('Number of Picks per Year by Warehouse_Section')
plt.xlabel('Year')
plt.ylabel('Number of Picks')
plt.legend(title='Warehouse_Section')
plt.show()

## ------ End of 'Number of Picks per Year by Warehouse_Section (Approach 1)' ------ ##


## ------ Monthly and Quarterly Analysis of the No. of Picks per WH Section (Approach 2) ------ ##
df2_transformed['Year'] = df2_transformed['Date'].dt.year
df2_transformed['Month'] = df2_transformed['Date'].dt.month # For monthly aggregation
df2_transformed['Quarter'] = df2_transformed['Date'].dt.to_period('Q') # For quarterly aggregation
print(df2_transformed.info())

monthly_picks = df2_transformed.groupby(['Year', 'Month', 'Warehouse_Section']).size().reset_index(name='Picks')
print(monthly_picks.head())

quarterly_picks = df2_transformed.groupby(['Year', 'Quarter', 'Warehouse_Section']).size().reset_index(name='Picks')
print(quarterly_picks.head())

#monthly_pivot = monthly_picks.pivot_table(index=['Year', 'Month'], columns='Warehouse_Section', values='Picks', fill_value=0)
#quarterly_pivot = quarterly_picks.pivot_table(index=['Year', 'Quarter'], columns='Warehouse_Section', values='Picks', fill_value=0)

## ------ End of 'Approach 2' ------ ##


# Converting the df2_transformed DF to a CSV file
# df2_transformed.to_csv("pick_data_transformed.csv", index=False)


## ----- Merging the Datasets ----- ##
merged_df = pd.merge(df2_transformed, df, on='SKU', how='left') # A left join includes all rows from the left DataFrame (df2_transformed), and matched rows from the right DataFrame (df). If there's no match, NaN values are filled in for columns from the second datframe, df.
print(merged_df.isnull().sum())
null_product_group_rows = merged_df[merged_df['Product_Group'].isnull()] # Show rows with Product_Group being null
print(null_product_group_rows)

## ----- End of 'Merging the Datasets' ----- ##


## ----- Creating a new DF called 'unique_order_details' ----- ##

# Grouping by 'Unique_Order_No' and getting the first and last timestamp for each order
unique_order_details = df2_transformed.groupby('Unique_Order_No').agg(
    Start_Time=('Date', 'first'),  # First timestamp for each order
    End_Time=('Date', 'last')      # Last timestamp for each order
).reset_index()

## ++ Enter your code here! ++ ##
# Add a new column with Unique warehouse sections in 2 steps
# Step 1: Aggregate unique Warehouse_Sections for each Unique_Order_No 7m 59.s
unique_warehouse_sections = (
    df2_transformed.groupby('Unique_Order_No')['Warehouse_Section']
    .agg(lambda x: ', '.join(x.unique()))  # Get unique Warehouse_Sections as a comma-separated string
)
# Step 2: Add the new column to unique_order_details
unique_order_details['Unique_Warehouse_Sections'] = unique_order_details['Unique_Order_No'].map(unique_warehouse_sections)

# Add a new column with pick count per order in 2 steps
# Step 1: Count the number of rows per Unique_Order_No in pick_data
order_row_counts = df2_transformed.groupby('Unique_Order_No').size()

# Step 2: Add the new column to unique_order_details
unique_order_details['Pick_Count'] = unique_order_details['Unique_Order_No'].map(order_row_counts)

# Add a new column that counts the number of unique warehouses per order
# Step 1: Count the number of unique warehouse sections in each order
unique_order_details['Unique_Warehouse_Count'] = unique_order_details['Unique_Warehouse_Sections'].str.split(', ').str.len().fillna(0).astype(int)

# Add a new column with Unique SKUs in 2 steps
# Step 1: Aggregate unique SKUs for each Unique_Order_No 8m 32.9s
unique_skus = (
    df2_transformed.groupby('Unique_Order_No')['SKU']
    .agg(lambda x: ', '.join(x.unique()))  # Get unique SKUs as a comma-separated string
)
# Step 2: Add the new column to unique_order_details
unique_order_details['Unique_SKUs'] = unique_order_details['Unique_Order_No'].map(unique_skus)

# Add a new column that counts the number of unique SKUs per order
# Step 1: Count the number of unique SKUs in each order
unique_order_details['Unique_SKU_Count'] = unique_order_details['Unique_SKUs'].str.split(', ').str.len().fillna(0).astype(int)

## ++ Enter your code here! ++ ##

## ----- End of 'unique_order_details DF' ----- ##


## ----- Product_Group Correlations ----- ##
# pip install -U mlxtend (to update the mlxtend library)

# Step 1: Sample 100,000 unique orders
subset_df = merged_df[merged_df['Unique_Order_No'].isin(
    merged_df['Unique_Order_No'].drop_duplicates().sample(100000, random_state=42)
)]

# Step 2: Remove duplicates of (Order Number, Product_Group) pairs
subset_df = subset_df.drop_duplicates(subset=['Unique_Order_No', 'Product_Group'])
print(subset_df.head())

# Step 3: One-hot encode the Product_Group per Order Number
one_hot_encoded_df = subset_df.groupby('Unique_Order_No')['Product_Group'].apply(lambda x: pd.Series(1, index=x)).unstack(fill_value=0) ## took approx 45 mins to execute
print(one_hot_encoded_df.head())

# Step 4: Run Apriori and association rules
from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(one_hot_encoded_df, min_support=0.01, use_colnames=True)
# rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0) # ideally, we should be using this but didn't work
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=len(frequent_itemsets)) # compatible with the old mlxtend version

# Display the rules (for quick testing, just the first few rows)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

## ----- End of 'Product_Group Correlations' ----- ##


## ------ EDA Univariate Analysis ------ ## 

# Analyzing/visualizing the dataset by taking one variable at a time

# Separating Numerical and Categorical variables for easy analysis
cat_cols = df2_transformed.select_dtypes(include=['object']).columns
num_cols = df2_transformed.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)

# Doing a Univariate Analysis using Histogram and Box Plot for continuous variables
for col in num_cols:
    print(col)
    print('Skew :', round(df2_transformed[col].skew(), 2))
    plt.figure(figsize = (15, 4))
    plt.subplot(1, 2, 1)
    df2_transformed[col].hist(grid=False)
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df2_transformed[col])
    plt.show()

# Categorical Variables are being visualized using a count plot
fig, axes = plt.subplots(1, 2, figsize = (18, 18))
fig.suptitle('Bar plot for all categorical variables in the dataset')
sns.countplot(ax=axes[0], x = 'Warehouse_Section', data = df2_transformed, color = 'blue', order = df2_transformed['Warehouse_Section'].value_counts().index)
sns.countplot(ax=axes[1], x = 'Unit', data = df2_transformed, color = 'blue', order = df2_transformed['Unit'].value_counts().index)
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

# log_transform(df2_transformed,['Position_in_Order','Pick_Volume'])


# extra comment

# A brand new commit 