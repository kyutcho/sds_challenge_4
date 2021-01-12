# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import sys
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer

# %%
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# train_data = pd.read_csv("data/public_listings.csv")

# Columns with mixed type
# train_data.columns[[15, 24, 33, 38, 42, 43, 51]]
#
# train_data[train_data.columns[[15, 24, 33, 38, 42, 43, 51]]]

# %%
# Row 17613 causes a problem - Read csv file again
train_raw = pd.read_csv("data/public_listings.csv", skiprows=[17614],
                        parse_dates=["host_since", "first_review", "last_review"])

# %%
# Make copy
train_data = train_raw.copy()

# %%
# Check the first 3 rows
train_data.head(3)

# %%
# Columns
train_data.columns

# %%
# Shape
train_data.shape

# %%
# Basic info
train_data.info()

# Duplicate rows
# %%
train_data[train_data.duplicated()]

# Missing values
# %%
# Function for missing values
def calc_missing(df):
    # total = df.isnull().sum().sort_values(ascending=False)
    # pct = round((df.isnull().sum() / len(df) * 100), 2).sort_values(ascending=False)

    total = df.isnull().sum()
    pct = round((df.isnull().sum() / len(df) * 100), 2)
    dtypes = df.dtypes

    missing_df = pd.concat([total, pct, dtypes], axis=1, keys=["Total", "Percent", "Dtypes"]) \
        .sort_values(by=["Total"], ascending=False).reset_index()
    missing_df = missing_df[missing_df["Total"] > 0]

    print(missing_df)


calc_missing(train_data)

# %%
# Delete instances that most features are missing
train_data.dropna(thresh=9, axis=0, inplace=True)

# %%
# One of these columns should be dropped
train_data["host_listings_count"].equals(train_data["host_total_listings_count"])

# %%
# Columns that need to be excluded
excl_cols = ["name", "description", "neighborhood_overview", "picture_url", "host_name",
             "host_location", "host_about", "host_thumbnail_url", "host_picture_url", "host_neighbourhood",
             "neighbourhood", "host_total_listings_count", "calendar_last_scraped"]
# train_data.loc[:, excl_cols]

train_data.drop(excl_cols, axis=1, inplace=True)

# %%
# Data Transformation (price)
train_data["price"] = train_data["price"].str.replace(',', '').astype(float)

# %%
# Data Transformation (Binary)
for var in ['has_availability', 'instant_bookable', 'host_is_superhost', 'host_has_profile_pic',
            'host_identity_verified']:
    train_data[var] = train_data[var].map({'t': 1, 'f': 0})

# %%
# Data Transformation (Percent)
train_data["host_response_rate"] = train_data["host_response_rate"].str.replace('%', '').astype(float)
train_data["host_acceptance_rate"] = train_data["host_acceptance_rate"].str.replace('%', '').astype(float)

# Imputing
# %% Categorical vars Manual Impute
train_data.fillna({"host_response_time": "no response data"}, inplace=True)
train_data.fillna({"host_verifications": "", "amenities": ""}, inplace=True)

# %%
# KNN Imputer
knn_train_copy = train_data.select_dtypes(include="number").copy(deep=True)
knn_imputer = KNNImputer(n_neighbors=2, weights="uniform")
knn_train_imputed = knn_imputer.fit_transform(knn_train_copy)

# Feature Engineering
# %%
# host_since
train_data["days_since_host"] = (train_data["host_since"].max() - train_data["host_since"]).dt.days
train_data.drop(["host_since"], axis=1, inplace=True)

# %%
# last_review - first_review
train_data["review_days_diff"] = (train_data["last_review"] - train_data["first_review"]).dt.days
train_data.drop(["last_review", "first_review"], axis=1, inplace=True)

# %%
# number of host_verification
train_data["num_host_verifications"] = train_data["host_verifications"].str.split(", ").apply(lambda x: len(x))
train_data.drop(["host_verifications"], axis=1, inplace=True)

# %%
# number of amenities
train_data["num_amenities"] = train_data["amenities"].str.split(", ").apply(lambda x: len(x) if x else 0)
train_data.drop(["amenities"], axis=1, inplace=True)

# %%
# num_bathroom, bath_is_private, bath_is_shared
train_data["bathrooms_text"] = train_data["bathrooms_text"].str.replace("half-bath", "0.5", case=False)
train_data["num_baths"] = train_data.bathrooms_text.str.extract('(\d+\.?\d*)')
train_data["bath_is_private"] = train_data["bathrooms_text"].str.contains("private", case=False)
train_data["bath_is_shared"] = train_data["bathrooms_text"].str.contains("shared", case=False)
train_data.drop(["bathrooms_text"], axis=1, inplace=True)

# %%
# Relationship between neighbourhood vs neighbourhood_group
ctab = pd.crosstab(train_data["neighbourhood_cleansed"], train_data["neighbourhood_group_cleansed"])
ctab_chk = ctab.apply(lambda x: ((x != 0).sum()), axis=1)
ctab_chk[ctab_chk == 0]

# This confirms all neighbourhood is assigned to one neighbourhood_group
train_data.drop(["neighbourhood_cleansed"], axis=1, inplace=True)

# %%
# Relationship between property_type vs room_type
ctab_2 = pd.crosstab(train_data["property_type"], train_data["room_type"]) \
    .sort_values(by=list(train_data["room_type"].unique()), ascending=False)
ctab_chk_2 = ctab_2.apply(lambda x: ((x != 0).sum()), axis=1)

# Verified overall property_type is well assigned to room_type -> Drop property_type
train_data.drop(["property_type"], axis=1, inplace=True)

# %%
# Beds per bedrooms
train_data["beds_per_rooms"] = train_data["beds"] / train_data["bedrooms"]

# EDA (price)
# %%
train_data["price"].describe()


# %%
def calc_IQR(col):
    return col.quantile(0.75) - col.quantile(0.25)


def calc_outlier(df, col):
    IQR = calc_IQR(df[col])

    third_qt = df[col].quantile(0.75)
    first_qt = df[col].quantile(0.25)

    outliers = df.loc[(df[col] <= (first_qt - 1.5 * IQR)) | (df[col] >= (third_qt + 1.5 * IQR)), col]

    n_outliers = len(outliers)

    return outliers, n_outliers


# %%
# number of outliers
price_outliers, price_n_outliers = calc_outlier(train_data, "price")


# %%
def df_no_outlier(df, col):
    IQR = calc_IQR(df[col])

    third_qt = df[col].quantile(0.75)
    first_qt = df[col].quantile(0.25)

    non_outlier_df = df.loc[(df[col] > (first_qt - 1.5 * IQR)) & (df[col] < (third_qt + 1.5 * IQR))]

    return non_outlier_df


# %%
fig = plt.figure()
ax = fig.add_subplot(111)
sns.distplot(train_data["price"], ax=ax)
ax.set(xlim=(0, 2000))
plt.show()

# %%
# log(price)
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.suptitle("This is the super title")
train_data["log_price"] = np.log(train_data["price"] + 0.00000000001)
sns.distplot(train_data["log_price"], ax=ax[1, 1])
ax[1, 1].set(title="This is title")
plt.show()

# %%
# sqrt(price)
train_data["sqrt_price"] = np.sqrt(train_data["price"])
sns.distplot(train_data["sqrt_price"])
plt.show()

# %%
sns.catplot(data=train_data, y="price", kind="box")

# %%
# sns.scatterplot(x = train_data["longitude"], y = train_data["latitude"])
# plt.scatter(x = train_data["longitude"], y = train_data["latitude"], alpha = 0.4)
train_data.loc[train_data["price"] < 304] \
    .plot(kind="scatter",
          x="longitude",
          y="latitude",
          alpha=0.4,
          c="price",
          cmap=plt.get_cmap("jet"),
          colorbar=True)
plt.show()

# %%
price_outliers_removed = df_no_outlier(train_data, "price")

# %%
sns.catplot(kind="box", data=train_data, x="neighbourhood_group_cleansed", y="price")

# %%
sns.relplot(kind="scatter", y=train_data["price"], x=train_data["beds"])

# %%
corr_df = train_data.select_dtypes("number").corr().abs()
mask = np.triu(np.ones_like(corr_df, dtype=bool))
tri_df = corr_df.mask(mask)
corr_filtered_df = train_data[[c for c in tri_df.columns if any(tri_df[c] > 0.8)]].corr()
sns.heatmap(corr_filtered_df, cmap="YlGnBu", annot=True)
