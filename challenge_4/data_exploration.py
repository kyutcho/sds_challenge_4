import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# train_data = pd.read_csv("data/public_listings.csv")

# Columns with mixed type
# train_data.columns[[15, 24, 33, 38, 42, 43, 51]]
#
# train_data[train_data.columns[[15, 24, 33, 38, 42, 43, 51]]]

# Row 17613 causes a problem - Read csv file again
train_data = pd.read_csv("data/public_listings.csv", skiprows=[17614], parse_dates=["first_review", "last_review"])

# Check the first 3 rows
train_data.head(3)

# Columns
train_data.columns

# Shape
train_data.shape

# Basic info
train_data.info()

# Function for missing values
def calc_missing(df):
    total = df.isnull().sum().sort_values(ascending=False)
    pct = round((df.isnull().sum() / len(df) * 100), 2).sort_values(ascending=False)

    missing_df = pd.concat([total, pct], axis=1, keys=["Total", "Percent"]).reset_index()

    return missing_df

missing_data = calc_missing(train_data)

# Missing values in host
train_data.loc[:, list(missing_data.iloc[:4, 0])]

# Fill missing values for host_response_rate and host_acceptance_rate
train_data["host_response_rate"] = train_data["host_response_rate"].str.split('%').str[0].fillna('0').astype("int")
train_data["host_acceptance_rate"] = train_data["host_acceptance_rate"].str.split('%').str[0].fillna('0').astype("int")

# Missing values in reviews
train_data.loc[:, list(missing_data.iloc[6:16, 0])]

# Fill missing values for review_scores
review_cols = [col for col in train_data.columns if "review_scores" in col]
review_cols.append("reviews_per_month")

train_data[review_cols] = train_data[review_cols].fillna(0)

# Missing values in neighborhood
train_data.loc[:, list(missing_data.iloc[4:6, 0])]
train_data["host_neighbourhood"]

# See columns that contain neighbo(u)r
train_data.filter(regex="neighb")

# Drop uncleaned columns
train_data.drop(["host_neighbourhood", "neighbourhood"], axis=1, inplace=True)

# Missing values in bedroom
train_data["bedrooms"]

print(calc_missing(train_data))