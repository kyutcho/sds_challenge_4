# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
# from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

# Feature Engineering
# %%
# host_since
train_data["days_since_host"] = (train_data["host_since"].max() - train_data["host_since"]).dt.days
# train_data.drop(["host_since"], axis=1, inplace=True)

# %%
# last_review - first_review
train_data["review_days_diff"] = (train_data["last_review"] - train_data["first_review"]).dt.days
# train_data.drop(["last_review", "first_review"], axis=1, inplace=True)

# %%
# number of host_verification
train_data["num_host_verifications"] = train_data["host_verifications"].str.count(",").apply(lambda x: x + 1)
train_data["num_host_verifications"].fillna(0, inplace=True)
# train_data.drop(["host_verifications"], axis=1, inplace=True)

# %%
# number of amenities
train_data["num_amenities"] = train_data["amenities"].str.count(",").apply(lambda x: x + 1)
train_data["num_amenities"].fillna(0, inplace=True)
# train_data.drop(["amenities"], axis=1, inplace=True)

# %%
# num_bathroom, bath_is_private, bath_is_shared
train_data["bathrooms_text"] = train_data["bathrooms_text"].str.replace("half-bath", "0.5", case=False)
train_data["num_baths"] = train_data.bathrooms_text.str.extract('(\d+\.?\d*)').astype(float)
train_data["bath_is_private"] = train_data["bathrooms_text"].str.contains("private", case=False)
train_data["bath_is_shared"] = train_data["bathrooms_text"].str.contains("shared", case=False)
# train_data.drop(["bathrooms_text"], axis=1, inplace=True)

# Price
# %%
train_data["price"].describe()

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 8))
sns.distplot(train_data["price"], ax=ax[0])
ax[0].set(xlim=(0, 2000))
sns.boxplot(data=train_data, y="price", ax=ax[1])
ax[1].set(ylim=(0, 2000))
plt.show()

# %%
# log(price)
epsilon = 10 ** -7
fig, ax = plt.subplots()
sns.distplot(np.log(train_data["price"] + epsilon), ax=ax, bins=50)
ax.set(title="Distribution with log(price)")
plt.show()

# %%
# sqrt(price)
fig, ax = plt.subplots()
# train_data["log_price"] = np.log(train_data["price"] + 0.00000000001)
sns.distplot(np.sqrt(train_data["price"]), ax=ax)
ax.set(title="Distribution with sqrt(price)")
plt.show()

# %%
# sns.scatterplot(x = train_data["longitude"], y = train_data["latitude"])
# plt.scatter(x = train_data["longitude"], y = train_data["latitude"], alpha = 0.4)
train_data.loc[train_data["price"] < 250] \
    .plot(kind="scatter",
          x="longitude",
          y="latitude",
          alpha=0.4,
          c="price",
          cmap=plt.get_cmap("jet"),
          colorbar=True)
plt.show()

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
price_outliers_removed = df_no_outlier(train_data, "price")

# %%
train_data.groupby(["neighbourhood_group_cleansed"]).agg({"price": ["mean", "median", "std"]}).reset_index()

# %%
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(121)
sns.boxplot(data=train_data, x="neighbourhood_group_cleansed", y="price", ax=ax)
ax.set(ylim=(0, 400))
ax = fig.add_subplot(122)
sns.boxplot(data=train_data, x="neighbourhood_group_cleansed", y="price", ax=ax)
plt.show()

# %%
# Potential outliars
outliers = train_data[train_data["price"] > 1500] \
    [["price", "neighbourhood_group_cleansed", "room_type", "beds", "bedrooms", "num_amenities"]] \
    .sort_values("price")
pd.pivot_table(outliers, values="price", index="room_type", columns="neighbourhood_group_cleansed",
               aggfunc=["median", "mean", "count"])

# %%
# Manhattan
fig, ax = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle("Price of Manhattan area")
sns.distplot(train_data[train_data["neighbourhood_group_cleansed"] == "Manhattan"][["price"]], bins=100, kde=True,
             ax=ax[0])
ax[0].set(xlim=(0, 2000))
sns.boxplot(data=train_data[train_data["neighbourhood_group_cleansed"] == "Manhattan"][["price"]], y="price", ax=ax[1])
ax[1].set(ylim=(0, 2000))
plt.show()

# %%
# Queens
fig, ax = plt.subplots(1, 2, figsize=(12, 8))
fig.suptitle("Price of Brooklyn area")
sns.distplot(train_data[train_data["neighbourhood_group_cleansed"] == "Brooklyn"][["price"]], bins=100, kde=True,
             ax=ax[0])
ax[0].set(xlim=(0, 2000))
sns.boxplot(data=train_data[train_data["neighbourhood_group_cleansed"] == "Brooklyn"][["price"]], y="price", ax=ax[1])
ax[1].set(ylim=(0, 2000))
plt.show()

# %%
train_data.groupby(["host_is_superhost"]).agg({"price": ["mean", "median", "std"]}).reset_index()

# %%
fig, ax = plt.subplots()
sns.boxplot(data=train_data, x="host_is_superhost", y="price", ax=ax)
ax.set_ylim(0, 500)
plt.show()

# %%
sns.scatterplot(y=train_data["price"], x=train_data["beds"])
plt.show()

# %%
sns.scatterplot(y=train_data["price"], x=train_data["bedrooms"])
plt.show()

# %%
pd.crosstab(train_data["bedrooms"], train_data["room_type"])

# %%
# Correlation
train_data.corr().loc[:, "price"].sort_values(ascending=False)

# %%
corr_mat = train_data.corr()
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr_mat, ax=ax, vmax=.8)
plt.show()

# %%
k = 10
cols = corr_mat.abs().nlargest(10, "price")["price"].index
corr_mat_zoomed = train_data[cols].corr()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()
sns.heatmap(corr_mat_zoomed, fmt=".2f", annot=True, cmap="YlGnBu", cbar=True, linewidths=.5, ax=ax)
plt.show()


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
calc_missing(train_data)

# %%
# One of these columns should be dropped
train_data["host_listings_count"].equals(train_data["host_total_listings_count"])

# %%
# Columns that need to be excluded
transf_cols = ["host_since", "last_review", "first_review", "host_verifications", "amenities", "bathrooms_text"]
irr_cols = ["name", "description", "neighborhood_overview", "picture_url", "host_name",
            "host_location", "host_about", "host_thumbnail_url", "host_picture_url", "host_neighbourhood",
            "neighbourhood", "host_total_listings_count", "calendar_last_scraped"]

train_data.drop(transf_cols + irr_cols, axis=1, inplace=True)

# %%
# Relationship between neighbourhood vs neighbourhood_group
ctab = pd.crosstab(train_data["neighbourhood_cleansed"], train_data["neighbourhood_group_cleansed"])
ctab_chk = ctab.apply(lambda x: ((x != 0).sum()), axis=1)
ctab_chk[ctab_chk != 1]

# This confirms all neighbourhood is assigned to one neighbourhood_group
train_data.drop(["neighbourhood_cleansed"], axis=1, inplace=True)

# %%
# Relationship between property_type vs room_type
ctab_2 = pd.crosstab(train_data["property_type"], train_data["room_type"])
ctab_chk_2 = ctab_2.apply(lambda x: ((x != 0).sum()), axis=1)
ctab_chk_2[ctab_chk_2 != 1]

# Verified overall property_type is well assigned to room_type -> Drop property_type
train_data.drop(["property_type"], axis=1, inplace=True)

# Dropping if less than 0.05% -- loses 21 rows (which is fine)
# %%
d_cols = round((train_data.isnull().sum() / len(train_data) * 100), 2).sort_values().index[:35]

train_data.dropna(subset=d_cols, inplace=True)

# %%
X_train = train_data.drop(columns=["price"], axis=1)
y_train = train_data["price"]

# Imputing
# %%
# Categorical vars - Imputing with new level (Missing) or Mode

# train_data["host_response_time"] = train_data["host_response_time"].fillna("Missing")

cat_pl = Pipeline([
    ('imputer', SimpleImputer(strategy="constant", fill_value="Missing")),
    ('oh_encoder', OneHotEncoder(drop="first"))])

# cat_pl.fit_transform(train_cat)x

# %%
# Numeric vars
X_train["host_response_rate"].fillna(0, inplace=True)
X_train["host_acceptance_rate"].fillna(
    X_train.groupby(["instant_bookable", "host_is_superhost"])["host_acceptance_rate"].transform("mean"),
    inplace=True)
X_train["beds"].fillna(X_train.groupby("accommodates")["beds"].transform("median"), inplace=True)
X_train["bedrooms"].fillna(X_train.groupby("accommodates")["bedrooms"].transform("median"), inplace=True)

num_pl = Pipeline([
    ('imputer', KNNImputer(n_neighbors=3, weights="uniform")),
    ('std_scaler', StandardScaler())
])


# %%
# Feature Engineering

# Beds per bedrooms
X_train["beds_per_rooms"] = X_train["beds"] / X_train["bedrooms"]

# %%
# Full Pipeline
train_cat_cols = X_train.select_dtypes("object").columns
train_num_cols = X_train.select_dtypes(exclude="object").columns
train_cat = X_train[train_cat_cols].values
train_num = X_train[train_num_cols].values

full_pl = ColumnTransformer([
    ("cat", cat_pl, train_cat_cols),
    ("num", num_pl, train_num_cols)
])

train_df_prepared = full_pl.fit_transform(X_train)


# %%
# KNN Imputer
# knn_train_copy = train_data.select_dtypes(include="number").copy(deep=True)
# knn_imputer = KNNImputer(n_neighbors=2, weights="uniform")
# knn_train_imputed = knn_imputer.fit_transform(knn_train_copy)

# %%
# Simple Imputer

# mean_train_copy = train_data.select_dtypes(include="number").copy(deep=True)
# mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# mean_train_imputed = mean_imputer.fit_transform(mean_train_copy)
# mean_train_imputed = pd.DataFrame(mean_train_imputed)
