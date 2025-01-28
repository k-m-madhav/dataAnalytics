import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, norm

# to ignore warnings
import warnings

warnings.filterwarnings("ignore")

# Write docstrings in triple quotation marks for all functions

# Load the CSV files
df = pd.read_csv(
    "002_product_data.csv",
    names=["SKU", "Description", "Product_Group"],
    header=None,
    encoding="latin1",
    engine="python",
    error_bad_lines=False,
    quoting=3,
)
df2 = pd.read_csv(
    "003_pick_data.csv",
    names=[
        "SKU",
        "Warehouse_Section",
        "Origin",
        "Order_No",
        "Position_in_Order",
        "Pick_Volume",
        "Unit",
        "Date",
    ],
    header=None,
    encoding="latin1",
    low_memory=False,
)

## ----- Handling Mixed Datatypes ----- ##
# print(df2['Warehouse_Section'].apply(type).value_counts()) O/P: <class 'str'>    33888990
df2[["Warehouse_Section", "Unit"]] = df2[["Warehouse_Section", "Unit"]].astype(
    "category"
)  # pandas' StringDtype ('string') is recommended for better memory efficiency and optimized string operations
df2["Date"] = pd.to_datetime(df2["Date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
df2[["SKU", "Order_No"]] = df2[["SKU", "Order_No"]].astype(
    "string"
)  # compare string and category dtype. category dtype might be the best. Use df2[''].nbytes to check the memory allocation
## ----- End of 'Handling Mixed Datatypes' ----- ##

## ----- SKU and Unit inconsistency ----- ##
# Group by 'SKU' and get unique 'Unit' values
unit_inconsistencies = df2.groupby("SKU")["Unit"].unique()

# Filter SKUs that have more than one unique 'Unit', sort the units and then create the summary with both unique units and their counts
inconsistent_skus_summary = pd.DataFrame(
    {
        "Unique_Units": unit_inconsistencies[
            unit_inconsistencies.apply(lambda x: len(x) > 1)
        ].apply(
            lambda x: sorted(list(x))
        ),  # Sort units to avoid discrepancies such as [Mt,St] and [St,Mt] being considered as two unique combinations
        "Count_of_Unique_Units": unit_inconsistencies[
            unit_inconsistencies.apply(lambda x: len(x) > 1)
        ].apply(
            lambda x: len(x)
        ),  # Count the unique units # All values are 2
    }
)
print(
    inconsistent_skus_summary["Unique Units"].value_counts()
)  # other unit combinations seem fine except [Mt,St] and it has a freq of 7

## ----- End of 'SKU and Unit inconsistency' ----- ##

## ----- Cleaning Product Data dataset ----- ##
print(
    df.isnull().sum()
)  # didn't find any null values because it considered "" i.e. Empty String, in the Description as a valid entry
print(df.nunique())  # SKU = 2199644, Description = 1422864, Product_Group = 18
df["Description"] = df["Description"].replace(
    '""', np.nan
)  # Replace exact `""` with NaN
df_transformed = df.copy(deep=True)

df_transformed["Description"] = df_transformed["Description"].str.strip(
    '"'
)  # Strip any leading or trailing double quotes from Description
df_transformed["Description"] = df_transformed["Description"].fillna("No Description")

# print(df2['SKU'].nunique() == df2['SKU'].str.lower().nunique())
df_transformed["Description_Cleaned"] = (
    df_transformed["Description"]
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.replace(r"[^\w\s]", "", regex=True)
)

# Step 1: Define a mapping for the German umlauts to their respective replacements
umlaut_mapping = {
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "ß": "ss",
    "Ä": "Ae",
    "Ö": "Oe",
    "Ü": "Ue",
}

# Step 2: Define a function to replace umlauts using the mapping
def replace_umlauts(text):
    for umlaut, replacement in umlaut_mapping.items():
        text = text.replace(umlaut, replacement)
    return text


# Step 3: Apply the function to both columns
df_transformed["Description_Cleaned"] = df_transformed["Description_Cleaned"].apply(
    replace_umlauts
)
df_transformed["Product_Group"] = df_transformed["Product_Group"].apply(replace_umlauts)

# print(df_transformed['Description_Cleaned'].nunique() - df_transformed['Description_Cleaned'].str.lower().nunique()) # 3193

descriptions_with_multiple_skus = df.groupby("Description")["SKU"].nunique()
descriptions_with_multiple_skus = descriptions_with_multiple_skus[
    descriptions_with_multiple_skus > 1
]

## ----- End of 'Cleaning Product Data dataset' ----- ##


## ----- Removing duplicate picks from Pick Data dataset ----- ##
print(df2.duplicated().sum())  # 8024
duplicate_picks = df2[df2.duplicated(keep=False)]
df2_transformed = df2.drop_duplicates()

## ----- End of 'Removing duplicate picks from Pick Data dataset' ----- ##


## ----- New Approach for Unique_Order_No ----- ##
# Finalized approach
def generate_unique_order_no(df, time_gap_threshold, large_gap_threshold_months=2):
    """
    This fn is to generate unique order no.s and the params are ...
    """
    # Step 1: Sort the dataframe by Order_No and Date
    df = df.sort_values(by=["Order_No", "Date"]).reset_index(drop=True)
    # Step 2: Precompute thresholds
    large_gap_threshold_seconds = (
        large_gap_threshold_months * 30.44 * 24 * 60 * 60
    )  # Convert months to seconds
    # Step 3: Calculate time differences using shift()
    df["Time_Difference"] = (df["Date"] - df["Date"].shift()).dt.total_seconds()
    # Reset Time_Difference to NaN where Order_No changes
    df["Time_Difference"] = df["Time_Difference"].where(
        df["Order_No"] == df["Order_No"].shift(), other=None
    )
    # Step 4: Initialize an empty list to store the Unique_Order_No for each row
    unique_order_no_list = pd.Series([None] * len(df), index=df.index)
    # Step 5: Group by Order_No for processing
    for order_no, group in df.groupby("Order_No", sort=False):
        # Initialize tracking variables
        unique_order_id = 1
        last_seen_rows = set()
        seen_positions = set()
        seen_skus = set()
        previous_origin = None

        for idx, (row_idx, row) in enumerate(group.iterrows()):
            if idx == 0:
                # First row always starts a new Unique_Order_No
                unique_order_no_list[row_idx] = f"{order_no}_v{unique_order_id}"
                last_seen_rows.add((row["Position_in_Order"], row["SKU"]))
                seen_positions.add(row["Position_in_Order"])
                seen_skus.add(row["SKU"])
                previous_origin = row["Origin"]
            else:
                # Fetch precomputed time gap
                time_gap = row["Time_Difference"]
                origin_changed = row["Origin"] != previous_origin
                large_gap = (
                    time_gap > large_gap_threshold_seconds
                    if not pd.isna(time_gap)
                    else False
                )
                long_gap = (
                    time_gap >= time_gap_threshold if not pd.isna(time_gap) else False
                )

                # Check SKU/Position logic
                pair_in_last_seen = (
                    row["Position_in_Order"],
                    row["SKU"],
                ) in last_seen_rows
                position_or_sku_in_last_seen = (
                    row["Position_in_Order"] in seen_positions
                    or row["SKU"] in seen_skus
                )

                # Logic to trigger a split
                if (
                    origin_changed
                    or large_gap
                    or (
                        long_gap
                        and not pair_in_last_seen
                        and position_or_sku_in_last_seen
                    )
                ):
                    unique_order_id += 1
                    last_seen_rows = set()
                    seen_positions = set()
                    seen_skus = set()

                # Update Unique_Order_No and tracking variables
                unique_order_no_list[row_idx] = f"{order_no}_v{unique_order_id}"
                last_seen_rows.add((row["Position_in_Order"], row["SKU"]))
                seen_positions.add(row["Position_in_Order"])
                seen_skus.add(row["SKU"])
                previous_origin = row["Origin"]

    # Step 6: Assign the generated Unique_Order_No values to the DataFrame
    df["Unique_Order_No"] = unique_order_no_list

    return df

df2_transformed = generate_unique_order_no(df2_transformed, 432000)

df2_transformed[["Unique_Order_No"]] = df2_transformed[["Unique_Order_No"]].astype(
    "string"
)
print(df2_transformed.nunique())
print(df2_transformed.info())

all_unique_order_nos = set(df2_transformed["Unique_Order_No"])  # 9371179
valid_unique_order_nos = set(
    df2_transformed.loc[df2_transformed["Position_in_Order"] == 1, "Unique_Order_No"]
)  # 9073280
invalid_unique_order_nos = all_unique_order_nos - valid_unique_order_nos  # 297899

# Sort by Unique_Order_No and Position_in_Order
df2_transformed = df2_transformed.sort_values(
    by=["Unique_Order_No", "Position_in_Order"]
)

# Renumber Position_in_Order efficiently
df2_transformed["Position_in_Order_Updated_Rank"] = (
    df2_transformed.groupby("Unique_Order_No")["Position_in_Order"]
    .rank(method="dense")
    .astype(int)
)  # 31.8 s to execute
df2_transformed = df2_transformed.sort_values(
    by=["Unique_Order_No", "Date"]
).reset_index(drop=True)

## ----- End of 'New Approach for Unique_Order_No' ----- ##

## ----- 0 Pick_Vol ----- ##

# Step 1: Create a flag for rows with Pick_Vol = 0 and Pick_Vol > 0
df2_transformed["Pick_Vol_Zero_Flag"] = df2_transformed["Pick_Volume"] == 0
df2_transformed["Pick_Vol_Positive_Flag"] = df2_transformed["Pick_Volume"] > 0

# Step 2: Group by Unique_Order_No, SKU, and Position_in_Order
grouped = df2_transformed.groupby(["Unique_Order_No", "SKU", "Position_in_Order"])

# Step 3: Check conditions within each group
matching_orders = grouped.agg(
    {
        "Pick_Vol_Zero_Flag": "any",  # Check if Pick_Vol = 0 exists
        "Pick_Vol_Positive_Flag": "any",  # Check if Pick_Vol > 0 exists
    }
).reset_index()

# Step 4: Filter groups that meet the criteria
matching_orders["Has_Zero_and_Positive_Pick_Vol"] = (
    matching_orders["Pick_Vol_Zero_Flag"] & matching_orders["Pick_Vol_Positive_Flag"]
)
matching_orders["Has_Zero_but_No_Positive_Pick_Vol"] = (
    matching_orders["Pick_Vol_Zero_Flag"] & ~matching_orders["Pick_Vol_Positive_Flag"]
)

# Step 5: Get the count of Unique_Order_No that meet the criteria and the 
# count of the ones that don't
orders_with_zero_and_positive = matching_orders[
    matching_orders["Has_Zero_and_Positive_Pick_Vol"]
][
    "Unique_Order_No"
].unique()  # 145,829
problematic_orders = matching_orders[
    matching_orders["Has_Zero_but_No_Positive_Pick_Vol"]
][
    "Unique_Order_No"
].unique()  # 2215
overlap_orders = set(orders_with_zero_and_positive).intersection(
    problematic_orders
)  # 76 Unique Orders come under both categories

# Analyze the strictly problematic orders
strictly_problematic_orders = list(set(problematic_orders) - overlap_orders)

# Filter all rows for strictly problematic orders
strictly_problematic_orders_df = df2_transformed[
    df2_transformed["Unique_Order_No"].isin(strictly_problematic_orders)
]

# Count unique combinations of Unique_Order_No, SKU and Position_in_Order with Pick_Vol = 0
zero_pick_stats = (
    strictly_problematic_orders_df[strictly_problematic_orders_df["Pick_Volume"] == 0]
    .groupby(["Unique_Order_No", "SKU", "Position_in_Order"])
    .size()
    .reset_index(name="Count")
)

# Count unique combinations of Warehouse_Section and Origin with Pick_Vol = 0
warehouse_and_origin_pick_stats = (
    strictly_problematic_orders_df[strictly_problematic_orders_df["Pick_Volume"] == 0]
    .groupby(["Warehouse_Section", "Origin"])
    .size()
    .reset_index(name="Count")
)

## ----- End of '0 Pick_Vol' ----- ##


print(
    df2_transformed.info()
)  # Gives info about the datatype of the variables (columns)
print(
    df2_transformed.nunique()
)  # Gives info on how many unique values are present for each column. ex: Warehouse_Section = 5, Origin = 2 and so on
print(
    df2_transformed["SKU"].value_counts()
)  # To see unique items of a column and their counts

# Is used to get the number of missing records in each column
print(df2_transformed.isnull().sum())
# There are no null values!

print(
    df2_transformed.describe().T
)  # Provide a statistics summary of data belonging to numerical datatype such as int, float

# 100 rows which have -ve Pick Vol.
print(df2_transformed[df2_transformed["Pick_Volume"] < 0])
print(df2_transformed[df2_transformed["Pick_Volume"] < 0].nunique())

# "Not really needed. Was doing some analysis to understand
# how many Order_No.s have multiple -ve Pick Vol"
# negative_pick_volume_df = df2_transformed[df2_transformed['Pick_Volume'] < 0]
# order_no_counts = negative_pick_volume_df['Order_No'].value_counts()
# non_unique_order_no = order_no_counts[order_no_counts > 1]
# print(non_unique_order_no)

## ------ Cleaning -ve Pick_Volume ------ ##
PICK_VOLUME = "Pick_Volume"
negative_indices = df2_transformed.index[
    df2_transformed[PICK_VOLUME] < 0
].to_list()  # storing all indices that have -ve Pick Vol in a list
print(len(negative_indices))

# All the indices which didn't match the "Condition" are stored in this list
missed_indices = []
# All the indices which matched the "Condition" but the sum of Pick_Volume equalled 0
missed_indices_sum_0 = []
# List to collect indices that need to be dropped
indices_to_drop = []
unique_order_nos_condition_failed = []
unique_order_nos_pick_volume_sum_0 = []
unique_order_nos_dropped = []

columns_to_drop_before_compare = df2_transformed.columns.difference(
    [PICK_VOLUME, "Pick_Vol_Positive_Flag", "Pick_Vol_Zero_Flag", "Time_Difference"]
)

for index in negative_indices:
    if index > 0:  # Ensure there's a row above to compare with
        row_above = df2_transformed.loc[
            index - 1, columns_to_drop_before_compare
        ]  # Row above the one with negative pick volume
        negative_row = df2_transformed.loc[
            index, columns_to_drop_before_compare
        ]  # Row with negative pick volume. Selecting only relevant columns for comparison

        # Compare rows excluding 'pick_volume'
        if row_above.equals(negative_row):
            combined_pick_volume = (
                df2_transformed.at[index - 1, PICK_VOLUME]
                + df2_transformed.at[index, PICK_VOLUME]
            )  # Combine the 'pick_volume' values

            if combined_pick_volume > 0:
                df2_transformed.at[
                    index - 1, PICK_VOLUME
                ] = combined_pick_volume  # Update the pick_volume for the row_above
                indices_to_drop.append(
                    index
                )  # Add the index of the negative pick_volume row to the drop list
                unique_order_nos_dropped.append(
                    df2_transformed.at[index, "Unique_Order_No"]
                )  # Store the Unique_Order_No for the row being dropped
            else:
                missed_indices_sum_0.append(
                    index
                )  # Track where combined Pick_Volume is <= 0
                unique_order_nos_pick_volume_sum_0.append(
                    df2_transformed.at[index, "Unique_Order_No"]
                )  # Store the Unique_Order_No for the row where combined Pick_Volume <= 0
        else:
            missed_indices.append(index)  # Track rows where they don't match
            unique_order_nos_condition_failed.append(
                df2_transformed.at[index, "Unique_Order_No"]
            )  # Store the Unique_Order_No where the 'Condition' failed

df2_transformed = df2_transformed.drop(indices_to_drop)  # Drop rows based on indices
df2_transformed = df2_transformed.reset_index(
    drop=True
)  # Reset index to create a clean, sequential index
print(df2_transformed.info())  # 71 rows less

# print(missed_indices) # count = 15
# print(df2_transformed.loc[31355201])
# print(df2_transformed[df2_transformed['Order_No'] == '03014512'])

# Converting the df2_transformed DF to a CSV file
# df2_transformed.to_csv("pick_data_transformed.csv", index=False)

## ------ End of 'Cleaning -ve Pick_Volume' ------ ##


## ------ Monthly and Quarterly Analysis of the No. of Picks per WH Section (Approach 2) ------ ##
df2_transformed["Year"] = df2_transformed["Date"].dt.year
df2_transformed["Month"] = df2_transformed["Date"].dt.month  # For monthly aggregation
df2_transformed["Quarter"] = df2_transformed["Date"].dt.to_period(
    "Q"
)  # For quarterly aggregation
print(df2_transformed.info())

monthly_picks = (
    df2_transformed.groupby(["Year", "Month", "Warehouse_Section"])
    .size()
    .reset_index(name="Picks")
)
print(monthly_picks.head())

quarterly_picks = (
    df2_transformed.groupby(["Year", "Quarter", "Warehouse_Section"])
    .size()
    .reset_index(name="Picks")
)
print(quarterly_picks.head())

# monthly_pivot = monthly_picks.pivot_table(index=['Year', 'Month'], columns='Warehouse_Section', values='Picks', fill_value=0)
# quarterly_pivot = quarterly_picks.pivot_table(index=['Year', 'Quarter'], columns='Warehouse_Section', values='Picks', fill_value=0)

## ------ End of 'Approach 2' ------ ##


## ----- Merging the Datasets ----- ##
merged_df = pd.merge(
    df2_transformed, df, on="SKU", how="left"
)  # A left join includes all rows from the left DataFrame (df2_transformed), and matched rows from the right DataFrame (df). If there's no match, NaN values are filled in for columns from the second datframe, df.
print(merged_df.isnull().sum())
null_product_group_rows = merged_df[
    merged_df["Product_Group"].isnull()
]  # Show rows with Product_Group being null
print(null_product_group_rows)

## ----- End of 'Merging the Datasets' ----- ##


## ----- Creating a new DF called 'unique_order_details' ----- ##

# Grouping by 'Unique_Order_No' and getting the first and last timestamp for each order
unique_order_details = (
    df2_transformed.groupby("Unique_Order_No")
    .agg(
        Start_Time=("Date", "first"),  # First timestamp for each order
        End_Time=("Date", "last"),  # Last timestamp for each order
    )
    .reset_index()
)

## ++ Enter your code here! ++ ##
# Add a new column with Unique warehouse sections in 2 steps
# Step 1: Aggregate unique Warehouse_Sections for each Unique_Order_No 7m 59.s
unique_warehouse_sections = df2_transformed.groupby("Unique_Order_No")[
    "Warehouse_Section"
].agg(
    lambda x: ", ".join(x.unique())
)  # Get unique Warehouse_Sections as a comma-separated string
# Step 2: Add the new column to unique_order_details
unique_order_details["Unique_Warehouse_Sections"] = unique_order_details[
    "Unique_Order_No"
].map(unique_warehouse_sections)

# Add a new column with pick count per order in 2 steps
# Step 1: Count the number of rows per Unique_Order_No in pick_data
order_row_counts = df2_transformed.groupby("Unique_Order_No").size()

# Step 2: Add the new column to unique_order_details
unique_order_details["Pick_Count"] = unique_order_details["Unique_Order_No"].map(
    order_row_counts
)

# Add a new column that counts the number of unique warehouses per order
# Step 1: Count the number of unique warehouse sections in each order
unique_order_details["Unique_Warehouse_Count"] = (
    unique_order_details["Unique_Warehouse_Sections"]
    .str.split(", ")
    .str.len()
    .fillna(0)
    .astype(int)
)

# Add a new column with Unique SKUs in 2 steps
# Step 1: Aggregate unique SKUs for each Unique_Order_No 8m 32.9s
unique_skus = df2_transformed.groupby("Unique_Order_No")["SKU"].agg(
    lambda x: ", ".join(x.unique())
)  # Get unique SKUs as a comma-separated string
# Step 2: Add the new column to unique_order_details
unique_order_details["Unique_SKUs"] = unique_order_details["Unique_Order_No"].map(
    unique_skus
)

# Add a new column that counts the number of unique SKUs per order
# Step 1: Count the number of unique SKUs in each order
unique_order_details["Unique_SKU_Count"] = (
    unique_order_details["Unique_SKUs"].str.split(", ").str.len().fillna(0).astype(int)
)

# Calculate time difference in seconds
unique_order_details["Time_Difference_Seconds"] = (
    unique_order_details["End_Time"] - unique_order_details["Start_Time"]
).dt.total_seconds()

# Map the Origin
origin_mapping = df2_transformed[["Unique_Order_No", "Origin"]].drop_duplicates()
unique_order_details = pd.merge(
    unique_order_details, origin_mapping, on="Unique_Order_No", how="left"
)

# Analysis
unique_order_details["Base_Order_No"] = unique_order_details[
    "Unique_Order_No"
].str.extract(r"([^\_]+)_v\d+")
# Step 2: Group by the base order number and sort within each group by Start_Time
unique_order_details.sort_values(by=["Base_Order_No", "Start_Time"], inplace=True)
unique_order_details["Next_Start_Time"] = unique_order_details.groupby("Base_Order_No")[
    "Start_Time"
].shift(-1)
unique_order_details["Time_Diff_Versions"] = (
    unique_order_details["Next_Start_Time"] - unique_order_details["End_Time"]
)

# Set threshold as 7 days
threshold_versions = pd.Timedelta(days=7)
# Filter based on the threshold: Only include differences less than 7 days
filtered_versions = unique_order_details[
    unique_order_details["Time_Diff_Versions"] < threshold_versions
]
print(df2_transformed[df2_transformed["Order_No"] == "01007725"])
print(df2_transformed[df2_transformed["Order_No"] == "54730981"])
print(df2_transformed[df2_transformed["Order_No"] == "09928757"])
print(df2_transformed[df2_transformed["Order_No"] == "09119721"])

## ++ Enter your code here! ++ ##

## ----- End of 'unique_order_details DF' ----- ##


## ------ EDA Univariate Analysis ------ ##

# Univariate Analysis of the no of Positions in a distinct Unique_Order_No
# Step 1: Extract max Position_in_Order_Updated (i.e. no of distinct SKUs) for each Unique_Order_No
order_max_positions = (
    df2_transformed.groupby(["Unique_Order_No", "Origin"])[
        "Position_in_Order_Updated_Rank"
    ]
    .max()
    .reset_index()
)
order_max_positions.rename(
    columns={"Position_in_Order_Updated_Rank": "Distinct_SKUs_per_Order"}, inplace=True
)

# Step 2: Stratify by Origin
store_orders = order_max_positions[order_max_positions["Origin"] == 46]
customer_orders = order_max_positions[order_max_positions["Origin"] == 48]

# Step 3: Basic statistics for each group
def summary_stats(group, column):
    stats = {
        "count": len(group),
        "mean": np.mean(group[column]),
        "median": np.median(group[column]),
        "mode": group[column].mode()[0] if not group[column].mode().empty else np.nan,
        "min": group[column].min(),
        "max": group[column].max(),
        "std": np.std(group[column]),
        "skew": skew(group[column]),
        "kurtosis": kurtosis(group[column]),
    }
    return stats


store_stats = summary_stats(store_orders, "Distinct_SKUs_per_Order")
customer_stats = summary_stats(customer_orders, "Distinct_SKUs_per_Order")
print("Store Orders Statistics:")
print(store_stats)
print("\nCustomer Orders Statistics:")
print(customer_stats)

# Plot histogram and normal distribution curve for store_orders
plt.figure(figsize=(8, 6))

# Store Orders Histogram
sns.histplot(
    store_orders["Distinct_SKUs_per_Order"],
    kde=False,
    color="skyblue",
    bins=20,
    stat="density",
)
# Fit normal distribution and plot the curve
mu_store, std_store = norm.fit(store_orders["Distinct_SKUs_per_Order"])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_store, std_store)
plt.plot(x, p, "r-", linewidth=2, label="Normal Curve")
plt.title("Histogram of Distinct SKUs per Store Order with Normal Curve (Origin 46)")
plt.xlabel("Distinct SKUs per Store Order")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.show()

# Plot histogram and normal distribution curve for customer_orders
plt.figure(figsize=(8, 6))

# Customer Orders Histogram
sns.histplot(
    customer_orders["Distinct_SKUs_per_Order"],
    kde=False,
    color="skyblue",
    bins=20,
    stat="density",
)
# Fit normal distribution and plot the curve
mu_customer, std_customer = norm.fit(customer_orders["Distinct_SKUs_per_Order"])
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu_customer, std_customer)
plt.plot(x, p, "r-", linewidth=2, label="Normal Curve")
plt.title("Histogram of Distinct SKUs per Customer Order with Normal Curve (Origin 48)")
plt.xlabel("Distinct SKUs per Customer Order")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.show()

# Function to compute modified z-score and flag outliers
def detect_outliers_modified_z(data, col_name, threshold=3.5):
    median = np.median(data[col_name])
    mad = np.median(np.abs(data[col_name] - median))

    # Handle edge case where MAD is zero
    if mad == 0:
        mad = 1e-6  # Small value to avoid division by zero

    # Calculate modified z-scores
    data["Modified_Z_Score"] = 0.6745 * (data[col_name] - median) / mad

    outlier_flag_col_name = col_name + "Outlier_Flag"

    # Flag outliers based on threshold
    data[outlier_flag_col_name] = np.abs(data["Modified_Z_Score"]) > threshold
    return data


# Apply outlier detection for Store and Customer Orders
store_orders = detect_outliers_modified_z(store_orders, "Distinct_SKUs_per_Order")
customer_orders = detect_outliers_modified_z(customer_orders, "Distinct_SKUs_per_Order")

# # Combine the results back into one DataFrame
# final_data = pd.concat([store_orders, customer_orders])

# # Reset index for cleanliness
# final_data.reset_index(drop=True, inplace=True)

# extra comment

# A brand new commit
