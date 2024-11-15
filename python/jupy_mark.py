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
df = pd.read_csv('002_product_data.csv', names=['SKU', 'Description', 'Product Group'], header=None, encoding='latin1', engine='python', error_bad_lines=False, quoting=3)
df2 = pd.read_csv('003_pick_data.csv', names=['SKU', 'Warehouse Section', 'Origin', 'Order No', 'Position in Order', 'Pick Volume', 'Unit', 'Date'], header=None, encoding='latin1', low_memory=False)

# %%
sorted_df2 = df2.sort_values(by=['Order No', 'Date'])
print(sorted_df2.head())

# %%
negative_pick_volume_df = sorted_df2[sorted_df2['Pick Volume'] < 0]
order_no_counts = negative_pick_volume_df['Order No'].value_counts()
non_unique_order_no = order_no_counts[order_no_counts > 1]
print(non_unique_order_no)

# %%
pick_volume = 'Pick Volume'
negative_indices = sorted_df2.index[sorted_df2[pick_volume] < 0].to_list()
print(len(negative_indices))

# %%
combined_rows = []
rows_to_combine = []
missed_indices = []
for index in negative_indices:
    if index > 0:
       row_above = sorted_df2.loc[index - 1]
       negative_row = sorted_df2.loc[index]

       if(row_above.drop(labels=pick_volume).equals(negative_row.drop(labels=pick_volume))):
        rows_to_combine.append((row_above, negative_row))
        combined_row = row_above.copy()
        combined_row[pick_volume] = row_above[pick_volume] + negative_row[pick_volume]
        sorted_df2.loc[index - 1] = combined_row
        combined_rows.append(combined_row)
       else:
          missed_indices.append(index)

# %%
negative_indices_removed = list(set(negative_indices) ^ set(missed_indices))
print(len(negative_indices_removed))

# %%
sorted_df2 = sorted_df2.drop(index=negative_indices_removed).reset_index(drop=True)

# %%
print(sorted_df2[sorted_df2['Pick Volume'] < 0].nunique())

# %%
sorted_df2['Date'] = pd.to_datetime(sorted_df2['Date'])

# %%
pick_data = sorted_df2.copy(deep=True)

# %%
print(pick_data.head())

# %%
# Calculate the time difference between consecutive rows within each Order Number group
pick_data['Time Difference'] = (pick_data['Date'] - pick_data['Date'].shift()).dt.days

# %%
pick_data['Time Difference'] = pick_data['Time Difference'].where(
    pick_data['Order No'] == pick_data['Order No'].shift()
)

# %%
print(pick_data[pick_data['Order No'] == '01000002'])

# %%
time_diffs = pick_data['Time Difference'].dropna()
print(time_diffs.value_counts().sort_index())

# %%
# Step 1: Plot distribution of time differences
plt.figure(figsize=(10, 6))
plt.hist(time_diffs, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Days Between Orders')
plt.ylabel('Frequency')
plt.title('Distribution of Time Differences Between Orders')
plt.show()

# %%
time_diffs_non_0_days = time_diffs[time_diffs != 0.0]

# %%
print(time_diffs_non_0_days.value_counts().sort_index())

# %%
# Step 1: Plot distribution of time differences
plt.figure(figsize=(10, 6))
plt.hist(time_diffs_non_0_days, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Days Between Orders')
plt.ylabel('Frequency')
plt.title('Distribution of Time Differences Between Orders (Excl 0)')
plt.show()

# %%
freq_non_0_days = time_diffs_non_0_days.value_counts().sort_index()

# %%
# Set the threshold in days
threshold_days = 5

# %%
# Create a flag where time difference is greater than threshold (indicating a new version)
pick_data['New Version Flag'] = (pick_data['Time Difference'] > threshold_days).fillna(False)

# %%
# Calculate the cumulative sum of the new version flag to create a version counter within each Order Number
pick_data['Version'] = pick_data.groupby('Order No')['New Version Flag'].cumsum() + 1

# %%
# Create the Unique Order No column using the Order Number and Version
pick_data['Unique Order No'] = pick_data['Order No'].astype(str) + "_v" + pick_data['Version'].astype(str)

# %%
print(pick_data[pick_data['Order No'] == '01000002'])

# %%
# Drop temporary columns if desired
pick_data = pick_data.drop(columns=['Time Difference', 'New Version Flag', 'Version'])

# %%
print(pick_data[pick_data['Order No'] == '01000002'])

# %%
pick_data = pick_data.sort_values(by=['Unique Order No', 'Date']).reset_index(drop=True)

# %%
print(pick_data.nunique())

# %%
pick_data['Year'] = pick_data['Date'].dt.year

# %%
diff_years = pick_data.groupby('Unique Order No')['Year'].nunique()
print(diff_years)

# %%
unique_order_no_with_diff_years = diff_years[diff_years > 1].index
print("Unique Order nuumbers with diff years: ", unique_order_no_with_diff_years)

# %%
print(pick_data[pick_data['Unique Order No'] == '02286155_v2'])

# %%
#pick_data.to_csv("pick_data_unique_order_no.csv", index=False)

# %%
merged_df = pd.merge(pick_data, df, on='SKU', how='left')

# %%
print(merged_df.isnull().sum())

# %%
# Show rows with Product Group being null
null_rows_product_group = merged_df[merged_df['Product Group'].isnull()]
print(null_rows_product_group)

# %%
print(df[df['SKU'] == 'Y91358'])

# %%
print(merged_df.nunique())

# %%
picks_per_product_group = merged_df['Product Group'].value_counts()

# %%
picks_per_product_group_df = pd.DataFrame(picks_per_product_group).reset_index()
picks_per_product_group_df.columns = ['Product Group', 'Overall Picks']
print(picks_per_product_group_df)

# %%
# Grouping by 'Unique Order No' and getting the first and last timestamp for each order
unique_order_details = pick_data.groupby('Unique Order No').agg(
    Start_Time=('Date', 'first'),  # First datetime for each order
    End_Time=('Date', 'last')      # Last datetime for each order
).reset_index()
print(unique_order_details.head())