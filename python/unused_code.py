# Cleaning Prod Dataset
# df_transformed['Description_Cleaned_Final'] = df_transformed['Description'].str.replace(r'\s+', ' ', regex=True).str.strip()
# df_transformed['Description_Cleaned_Final'] = df_transformed['Description_Cleaned_Final'].str.replace(r'[^\w\s]', '', regex=True)
# df_transformed['Description_Cleaned_Final'] = df_transformed['Description_Cleaned_Final'].apply(lambda x: ''.join(c for c in x if c.isprintable()))
# df_transformed['Description_Cleaned_Final'] = df_transformed['Description_Cleaned_Final'].apply(replace_umlauts)

# df_transformed['Description_Cleaned'] = df_transformed['Description_Cleaned'].apply(repr)
# df_transformed['Description_Cleaned'] = df_transformed['Description_Cleaned'].str.strip("'")

##  ------ Fix the Repeating Order Numbers by creating Unique Order Numbers
# based on the assumption that an order is assumed to be completed within 5 days ------ ##

# df2_transformed = df2_transformed.sort_values(by=["Order_No", "Date"]).reset_index(
#     drop=True
# )  # Sort by 'Order_No' and 'Date'

# # df2_transformed['Time_Difference'] = df2_transformed.groupby('Order_No')['Date'].diff().dt.days # Calculate the time difference between consecutive rows within each Order Number group (Took 37m 5.6s to execute)

# df2_transformed["Time_Difference"] = (
#     df2_transformed["Date"] - df2_transformed["Date"].shift()
# ).dt.days  # Fastest approach acc to chat-gpt to calculate time difference. Directly using shift on sorted data
# # Reset differences to NaN where Order_No changes
# df2_transformed["Time_Difference"] = df2_transformed["Time_Difference"].where(
#     df2_transformed["Order_No"] == df2_transformed["Order_No"].shift()
# )

# # print(df2_transformed[df2_transformed['Order_No'] == '01000002'])
# time_diffs = df2_transformed[
#     "Time_Difference"
# ].dropna()  # Drop NaN values (first occurrence in each group)
# print(time_diffs.value_counts().sort_index())
# time_diffs_non_0_days = time_diffs[
#     time_diffs != 0.0
# ]  # filters out all entries where the value is 0.0
# freq_non_0_days = time_diffs_non_0_days.value_counts().sort_index()

# # Plotting a distribution of time differences excl 0
# plt.figure(figsize=(10, 6))
# plt.hist(time_diffs_non_0_days, bins=10, color="skyblue", edgecolor="black")
# plt.xlabel("Days Between Orders")
# plt.ylabel("Frequency")
# plt.title("Distribution of Time Differences Between Orders (Excl 0)")
# plt.show()

# threshold_days = 5  # Set the threshold in days (to differentiate legitimate order repeats from duplicates, Chat-GPT recommends to assume that an order takes 5 days to complete)
# df2_transformed["New_Version_Flag"] = (
#     df2_transformed["Time_Difference"] > threshold_days
# ).fillna(
#     False
# )  # Create a flag where time difference is greater than threshold (indicating a new version)
# # print(df2_transformed[df2_transformed['Order_No'] == '01000002'])
# df2_transformed["Version"] = (
#     df2_transformed.groupby("Order_No")["New_Version_Flag"].cumsum() + 1
# )  # Calculate the cumulative sum of the new version flag to create a version counter within each Order Number
# df2_transformed["Unique_Order_No"] = (
#     df2_transformed["Order_No"].astype(str)
#     + "_v"
#     + df2_transformed["Version"].astype(str)
# )  # Creating the Unique_Order_No column using the Order Number and Version
# df2_transformed = df2_transformed.drop(
#     columns=["Time_Difference", "New_Version_Flag", "Version"]
# )  # Dropping the temporary columns
# # print(df2_transformed[df2_transformed['Order_No'] == '01000002'])
# df2_transformed = df2_transformed.sort_values(
#     by=["Unique_Order_No", "Date"]
# ).reset_index(
#     drop=True
# )  # Sort by 'Unique_Order_No' and 'Date'
# print(df2_transformed.nunique())

# Testing if an Edge Case is covered. Start Date is 2015-12-30 and End Date is 2016-01-03 which is within the 5 day threshold and needs to be considered as 1 Unique Order
# df2_transformed['Year'] = df2_transformed['Date'].dt.year
# diff_years = df2_transformed.groupby('Unique_Order_No')['Year'].nunique()
# unique_order_no_with_diff_years = diff_years[diff_years > 1].index
# print("Unique Order numbers with diff years: ", unique_order_no_with_diff_years)
# print(df2_transformed[df2_transformed['Unique_Order_No'] == '02286155_v2'])

## ------ End of 'Fix the Repeating Order_No by creating Unique_Order_No' ------ ##


# # To figure out the ideal long time_gap_threshold
# df2_test = df2_transformed.sort_values(by=['Order_No', 'Date'])
# df2_test['Time_Diff'] = df2_test['Date'].diff().dt.total_seconds()
# df2_test['Order_Change'] = df2_test['Order_No'] != df2_test['Order_No'].shift(1)
# time_diff_test = df2_test.loc[~df2_test['Order_Change'], 'Time_Diff']

# time_diff_log = np.log1p(time_diff_test.dropna())
# # Define lower and upper percentiles for outlier removal
# lower_percentile = time_diff_log.quantile(0.01)
# upper_percentile = time_diff_log.quantile(0.99)

# # Filter out data based on percentiles
# cleaned_data_log = time_diff_log[(time_diff_log >= lower_percentile) & (time_diff_log <= upper_percentile)]

# from scipy.stats import zscore
# # Apply Z-score on cleaned log-transformed data
# z_scores_log = zscore(cleaned_data_log)
# # Define threshold for outliers (e.g., Z-score > 3 or < -3)
# outliers_log = abs(z_scores_log) > 3
# cleaned_data_log_final = cleaned_data_log[~outliers_log]

# from sklearn.cluster import KMeans
# # Prepare data for clustering
# time_gaps_cleaned = cleaned_data_log_final.values.reshape(-1, 1)

# # Apply K-Means clustering
# kmeans = KMeans(n_clusters=2, random_state=42).fit(time_gaps_cleaned)
# centroids = kmeans.cluster_centers_

# # Identify threshold between clusters
# threshold_cluster_log = max(centroids.min(), centroids.max())

# # Reverse the log transformation (exp1 reverses log1p)
# threshold_cluster = np.expm1(threshold_cluster_log)
# print(f"Cluster-based Threshold: {threshold_cluster / (24 * 3600)} days")  # Convert to days O/P: 1041.7199234325633 days

# # Create the predicted labels (i.e., which cluster each point belongs to)
# labels = kmeans.labels_
# # Visualize the clustering result
# plt.scatter(time_gaps_cleaned, np.zeros_like(time_gaps_cleaned), c=labels, cmap='viridis', alpha=0.7)
# plt.scatter(centroids, np.zeros_like(centroids), c='red', marker='x', s=200, label="Centroids")
# plt.title("K-Means Clustering")
# plt.xlabel("Log-Transformed Time Difference")
# plt.legend()
# plt.show()


# # Renumber Position_in_Order efficiently
# df2_transformed['Position_in_Order_Updated'] = df2_transformed.groupby('Unique_Order_No')['Position_in_Order'].transform(
#     lambda x: pd.factorize(x)[0] + 1 # 35 mins to execute
# )


# Display the first few rows of the DataFrame
# print(df.head())
# print(df2.describe(include='all'))
# print(df2.describe())
# print(df2.head())
# To print all rows corresponding to the specified SKU
# from product_data.csv
# skus = ['A80704', '387126', '283136']
# print(df[df['SKU'].isin(skus)])


## ------ To get the Number of Picks per Year
# by Warehouse_Section (Approach 1) [Not the best] ------ ##
print(df2_transformed.info())

# Count picks per order number and warehouse section
pick_counts = (
    df2_transformed.groupby(["Order_No", "Warehouse_Section"])
    .size()
    .reset_index(name="pick_count")
)
print(pick_counts.head())
print(df2_transformed[df2_transformed["Order_No"] == "01000002"])

# Now aggregate the start and end timestamps
timestamp_agg = (
    df2_transformed.groupby(["Order_No", "Warehouse_Section"])
    .agg(start_timestamp=("Date", "min"), end_timestamp=("Date", "max"))
    .reset_index()
)
print(timestamp_agg.head())

# Merge the two DataFrames to get pick counts with timestamps
aggregated_df = pd.merge(
    timestamp_agg, pick_counts, on=["Order_No", "Warehouse_Section"]
)
print(aggregated_df.head())

# Extract the start year
aggregated_df["start_year"] = aggregated_df["start_timestamp"].dt.year

# Aggregate the counts by year and warehouse section
yearly_counts = (
    aggregated_df.groupby(["start_year", "Warehouse_Section"])
    .agg(
        order_count=(
            "pick_count",
            "sum",
        )  # Sum the pick counts for each year and warehouse section
    )
    .reset_index()
)
print(yearly_counts.head())

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(
    data=yearly_counts, x="start_year", y="order_count", hue="Warehouse_Section"
)
plt.title("Number of Picks per Year by Warehouse_Section")
plt.xlabel("Year")
plt.ylabel("Number of Picks")
plt.legend(title="Warehouse_Section")
plt.show()

## ------ End of 'Number of Picks per Year by Warehouse_Section (Approach 1)' ------ ##


## ----- Product_Group Correlations ----- ##
# pip install -U mlxtend (to update the mlxtend library)

# Step 1: Sample 100,000 unique orders
subset_df = merged_df[
    merged_df["Unique_Order_No"].isin(
        merged_df["Unique_Order_No"].drop_duplicates().sample(100000, random_state=42)
    )
]

# Step 2: Remove duplicates of (Order Number, Product_Group) pairs
subset_df = subset_df.drop_duplicates(subset=["Unique_Order_No", "Product_Group"])
print(subset_df.head())

# Step 3: One-hot encode the Product_Group per Order Number
one_hot_encoded_df = (
    subset_df.groupby("Unique_Order_No")["Product_Group"]
    .apply(lambda x: pd.Series(1, index=x))
    .unstack(fill_value=0)
)  ## took approx 45 mins to execute
print(one_hot_encoded_df.head())

# Step 4: Run Apriori and association rules
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(one_hot_encoded_df, min_support=0.01, use_colnames=True)
# rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0) # ideally, we should be using this but didn't work
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0,
    num_itemsets=len(frequent_itemsets),
)  # compatible with the old mlxtend version

# Display the rules (for quick testing, just the first few rows)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head())

## ----- End of 'Product_Group Correlations' ----- ##


# Analyzing/visualizing the dataset by taking one variable at a time

# Separating Numerical and Categorical variables for easy analysis
# cat_cols = df2_transformed.select_dtypes(include=['object']).columns
# num_cols = df2_transformed.select_dtypes(include=np.number).columns.tolist()
# print("Categorical Variables:")
# print(cat_cols)
# print("Numerical Variables:")
# print(num_cols)

# # Doing a Univariate Analysis using Histogram and Box Plot for continuous variables
# for col in num_cols:
#     print(col)
#     print('Skew :', round(df2_transformed[col].skew(), 2))
#     plt.figure(figsize = (15, 4))
#     plt.subplot(1, 2, 1)
#     df2_transformed[col].hist(grid=False)
#     plt.ylabel('count')
#     plt.subplot(1, 2, 2)
#     sns.boxplot(x=df2_transformed[col])
#     plt.show()

# # Categorical Variables are being visualized using a count plot
# fig, axes = plt.subplots(1, 2, figsize = (18, 18))
# fig.suptitle('Bar plot for all categorical variables in the dataset')
# sns.countplot(ax=axes[0], x = 'Warehouse_Section', data = df2_transformed, color = 'blue', order = df2_transformed['Warehouse_Section'].value_counts().index)
# sns.countplot(ax=axes[1], x = 'Unit', data = df2_transformed, color = 'blue', order = df2_transformed['Unit'].value_counts().index)
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