# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# %%
# Load the CSV files
df = pd.read_csv('002_product_data.csv', names=['SKU', 'Description', 'Product_Group'], header=None, encoding='latin1', engine='python', error_bad_lines=False, quoting=3)
df2 = pd.read_csv('003_pick_data.csv', names=['SKU', 'Warehouse_Section', 'Origin', 'Order_No', 'Position_in_Order', 'Pick_Volume', 'Unit', 'Date'], header=None, encoding='latin1', low_memory=False)

# %%
## ----- Cleaning Product Data dataset ----- ##
print(df.isnull().sum()) # didn't find any null values because it considered "" i.e. Empty String in Description as a valid entry
print(df.nunique()) # SKU = 2199644, Description = 1422864, Product_Group = 18
df['Description'] = df['Description'].replace('""', np.nan) # Replace exact `""` with NaN
df['Description'] = df['Description'].str.strip('"') # Strip any leading or trailing double quotes from Description

descriptions_with_multiple_skus = df.groupby('Description')['SKU'].nunique()
descriptions_with_multiple_skus = descriptions_with_multiple_skus[descriptions_with_multiple_skus > 1]
## ----- End of 'Cleaning Product Data dataset' ----- ## 

# %%
## ----- Removing duplicate picks from Pick Data dataset ----- ##
print(df2.duplicated().sum())
duplicate_picks = df2[df2.duplicated(keep=False)]
df2_transformed = df2.drop_duplicates()
## ----- End of 'Removing duplicate picks from Pick Data dataset' ----- ##

# %%
##  ------ Fix the Repeating Order Numbers by creating Unique Order Numbers ------ ##
df2_transformed['Date'] = pd.to_datetime(df2_transformed['Date'])
df2_transformed = df2_transformed.sort_values(by=['Order_No', 'Date']).reset_index(drop=True) # Sort by 'Unique_Order_No' and 'Date'

# %%
# Calculate the time difference between consecutive rows within each Order Number group
df2_transformed['Time_Difference'] = (df2_transformed['Date'] - df2_transformed['Date'].shift()).dt.days

# %%
df2_transformed['Time_Difference'] = df2_transformed['Time_Difference'].where(
    df2_transformed['Order_No'] == df2_transformed['Order_No'].shift()
)

# %%
print(df2_transformed[df2_transformed['Order_No'] == '01000002'])

# %%
time_diffs = df2_transformed['Time Difference'].dropna()
print(time_diffs.value_counts().sort_index())
time_diffs_non_0_days = time_diffs[time_diffs != 0.0] # filters out all entries where the value is 0.0
freq_non_0_days = time_diffs_non_0_days.value_counts().sort_index()

# %%
# Plotting a distribution of time differences excl 0
plt.figure(figsize=(10, 6))
plt.hist(time_diffs_non_0_days, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Days Between Orders')
plt.ylabel('Frequency')
plt.title('Distribution of Time Differences Between Orders (Excl 0)')
plt.show()

# %%
threshold_days = 5 # Set the threshold in days (to differentiate legitimate order repeats from duplicates, Chat-GPT recommends to assume that an order takes 5 days to complete)
df2_transformed['New_Version_Flag'] = (df2_transformed['Time_Difference'] > threshold_days).fillna(False) # Create a flag where time difference is greater than threshold (indicating a new version)

# %%
df2_transformed['Version'] = df2_transformed.groupby('Order_No')['New_Version_Flag'].cumsum() + 1 # Calculate the cumulative sum of the new version flag to create a version counter within each Order Number

# %%
df2_transformed['Unique_Order_No'] = df2_transformed['Order_No'].astype(str) + "_v" + df2_transformed['Version'].astype(str) # Creating the Unique_Order_No column using the Order Number and Version
print(df2_transformed[df2_transformed['Order_No'] == '01000002'])

# %%
df2_transformed = df2_transformed.drop(columns=['Time_Difference', 'New_Version_Flag', 'Version']) # Dropping the temporary columns

# %%
df2_transformed = df2_transformed.sort_values(by=['Unique_Order_No', 'Date']).reset_index(drop=True) # Sort by 'Unique_Order_No' and 'Date'

# %%
# df2_transformed['Year'] = df2_transformed['Date'].dt.year
# diff_years = df2_transformed.groupby('Unique_Order_No')['Year'].nunique()
# unique_order_no_with_diff_years = diff_years[diff_years > 1].index
# print("Unique Order numbers with diff years: ", unique_order_no_with_diff_years)

# %%
# print(df2_transformed[df2_transformed['Unique_Order_No'] == '02286155_v2'])

## ------ End of 'Fix the Repeating Order_No by creating Unique_Order_No' ------ ##

# %%
## ----- 0 Pick_Vol ----- ##
# Step 1: Create a flag for rows with Pick_Vol = 0 and Pick_Vol > 0
df2_transformed['Pick_Vol_Zero_Flag'] = df2_transformed['Pick_Volume'] == 0
df2_transformed['Pick_Vol_Positive_Flag'] = df2_transformed['Pick_Volume'] > 0

# Step 2: Group by Unique_Order_No, SKU, and Position_in_Order
grouped = df2_transformed.groupby(['Unique_Order_No', 'SKU', 'Position_in_Order'])

# %%
# Step 3: Check conditions within each group
matching_orders = grouped.agg({
    'Pick_Vol_Zero_Flag': 'any',      # Check if Pick_Vol = 0 exists
    'Pick_Vol_Positive_Flag': 'any'   # Check if Pick_Vol > 0 exists
}).reset_index()

# Step 4: Filter groups that meet the criteria
matching_orders['Has_Zero_and_Positive_Pick_Vol'] = matching_orders['Pick_Vol_Zero_Flag'] & matching_orders['Pick_Vol_Positive_Flag']
matching_orders['Has_Zero_but_No_Positive_Pick_Vol'] = matching_orders['Pick_Vol_Zero_Flag'] & ~matching_orders['Pick_Vol_Positive_Flag']

# %%
# Step 5: Get the count of Unique_Order_No that meet the criteria and the count of the ones that don't
orders_with_zero_and_positive = matching_orders[matching_orders['Has_Zero_and_Positive_Pick_Vol']]['Unique_Order_No'].unique() # 145,829
problematic_orders = matching_orders[matching_orders['Has_Zero_but_No_Positive_Pick_Vol']]['Unique_Order_No'].unique() # 2215
overlap_orders = set(orders_with_zero_and_positive).intersection(problematic_orders) # 76 Unique Orders come under both categories

# %%
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

# %%
# negative_pick_volume_df = df2_transformed[df2_transformed['Pick_Volume'] < 0]
# order_no_counts = negative_pick_volume_df['Order_No'].value_counts()
# non_unique_order_no = order_no_counts[order_no_counts > 1]
# print(non_unique_order_no)

# %%
## ------ Cleaning -ve Pick_Volume ------ ##
pick_volume = 'Pick_Volume'
negative_indices = df2_transformed.index[df2_transformed[pick_volume] < 0].to_list() # storing all indices that have -ve Pick Vol in a list
print(len(negative_indices))

# %%
missed_indices = [] # All the indices which didn't match the "Condition" are stored in this list
missed_indices_sum_0 = [] # All the indices which matched the "Condition" but the sum of Pick_Volume equalled 0
indices_to_drop = [] # List to collect indices that need to be dropped
columns_to_drop_before_compare = df2_transformed.columns.difference([pick_volume, 'Pick_Positive_Flag', 'Pick_Zero_Flag'])

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
            else:
                missed_indices_sum_0.append(index) # Track where combined pick_volume is <= 0
        else:
            missed_indices.append(index) # Track rows where they don't match

# %%
df2_transformed = df2_transformed.drop(indices_to_drop) # Drop rows based on indices
df2_transformed = df2_transformed.reset_index(drop=True) # Reset index to create a clean, sequential index
print(df2_transformed[df2_transformed['Pick_Volume'] < 0].info()) # 71 rows less
## ------ End of 'Cleaning -ve Pick_Volume' ------ ##

# %%
print(df2_transformed[df2_transformed['Pick_Volume'] < 0].nunique())

# %%
#df2_transformed.to_csv("pick_data_transformed.csv", index=False)

# %%
merged_df = pd.merge(df2_transformed, df, on='SKU', how='left')
print(merged_df.isnull().sum())

# %%
# Show rows with Product_Group being null
null_rows_product_group = merged_df[merged_df['Product_Group'].isnull()]
print(null_rows_product_group)

# %%
print(df[df['SKU'] == 'Y91358'])

# %%
print(merged_df.nunique())

# %%
# picks_per_product_group = merged_df['Product_Group'].value_counts()
# picks_per_product_group_df = pd.DataFrame(picks_per_product_group).reset_index()
# picks_per_product_group_df.columns = ['Product_Group', 'Overall Picks']
# print(picks_per_product_group_df)

# %%
## ----- Creating a new DF called 'unique_order_details' ----- ##
# Grouping by 'Unique_Order_No' and getting the first and last timestamp for each order
unique_order_details = df2_transformed.groupby('Unique_Order_No').agg(
    Start_Time=('Date', 'first'),  # First datetime for each order
    End_Time=('Date', 'last')      # Last datetime for each order
).reset_index()
print(unique_order_details.head())

# %%
# Add a new column with Unique warehouse sections in 2 steps
# Step 1: Aggregate unique Warehouse_Sections for each Unique_Order_No
unique_warehouse_sections = (
    df2_transformed.groupby('Unique_Order_No')['Warehouse_Section']
    .unique()
    .apply(lambda x: ', '.join(sorted(x)))  # Convert to a comma-separated string
)
# Step 2: Add the new column to unique_order_details
unique_order_details['Unique_Warehouse_Sections'] = unique_order_details['Unique_Order_No'].map(unique_warehouse_sections)

# %%
# Add a new column with pick count per order in 2 steps
# Step 1: Count the number of rows per Unique_Order_No in pick_data
order_row_counts = df2_transformed.groupby('Unique_Order_No').size()

# Step 2: Add the new column to unique_order_details
unique_order_details['Pick_Count'] = unique_order_details['Unique_Order_No'].map(order_row_counts)

# %%
# Add a new column that counts the number of unique warehouses per order
# Step 1: Count the number of unique warehouse sections in each order
unique_order_details['Unique_Warehouse_Count'] = unique_order_details['Unique_Warehouse_Sections'].apply(
    lambda x: len(x.split(', ')) if isinstance(x, str) else 0
)

# %%
# Add a new column with Unique SKUs in 2 steps
# Step 1: Aggregate unique SKUs for each Unique_Order_No
unique_skus = (
    df2_transformed.groupby('Unique_Order_No')['SKU']
    .unique()
    .apply(lambda x: ', '.join(sorted(x)))  # Convert to a comma-separated string
)
# Step 2: Add the new column to unique_order_details
unique_order_details['Unique_SKUs'] = unique_order_details['Unique_Order_No'].map(unique_skus)

# %%
# Add a new column that counts the number of unique SKUs per order
# Step 1: Count the number of unique SKUs in each order
unique_order_details['Unique_SKU_Count'] = unique_order_details['Unique_SKUs'].apply(
    lambda x: len(x.split(', ')) if isinstance(x, str) else 0
)