#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import sys
from sklearn.impute import KNNImputer

#%%
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# train_data = pd.read_csv("data/public_listings.csv")

# Columns with mixed type
# train_data.columns[[15, 24, 33, 38, 42, 43, 51]]
#
# train_data[train_data.columns[[15, 24, 33, 38, 42, 43, 51]]]

#%% 
# Row 17613 causes a problem - Read csv file again
train_raw = pd.read_csv("data/public_listings.csv", skiprows=[17614], parse_dates=["first_review", "last_review"])

#%% 
# Make copy
train_data = train_raw.copy()

#%% 
# Check the first 3 rows
train_data.head(3)

#%% 
# Columns
train_data.columns

#%% 
# Shape
train_data.shape

#%% 
# Basic info
train_data.info()

#%% 
# Columns that need to be excluded
excl_cols = ["name", "description", "neighborhood_overview", "picture_url", "host_name",\
             "host_location", "host_about", "host_thumbnail_url", "host_picture_url"]
# train_data.loc[:, excl_cols]

train_data.drop(excl_cols, axis = 1, inplace = True)

#%% 
# Data Transformation (price)
train_data["price"] = train_data["price"].str.replace(',', '').astype(float)

#%% 
# Data Transformation (Binary)
for var in ['has_availability', 'instant_bookable', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified']:
    train_data[var] = train_data[var].map({'t':1, 'f':0})

#%%
# Data Transformation (Percent)
train_data["host_response_rate"] = train_data["host_response_rate"].str.replace('%', '').astype(float)
train_data["host_acceptance_rate"] = train_data["host_acceptance_rate"].str.replace('%', '').astype(float)

#%%
# EDA (price)
train_data["price"].describe()

#%%
ax = sns.distplot(train_data["price"])
# ax.set(xlim = (0,2000))
plt.show()

#%%
# log(price)
sns.distplot(np.log(train_data["price"]+0.00000000001))
plt.show()

#%%
# sqrt(price)
sns.distplot(np.sqrt(train_data["price"]+0.00000000001))
plt.show()

#%%
ax = sns.histplot(data = train_data, x = "price")
ax.set(xlim = (0,2000))
plt.show()

#%%
sns.catplot(data = train_data, y = "price", kind = "box")

#%%
sns.relplot(kind = "scatter", y = train_data["price"], x = train_data["beds"])

# Missing values
#%% 
# Function for missing values
def calc_missing(df): 
    total = df.isnull().sum().sort_values(ascending=False)
    pct = round((df.isnull().sum() / len(df) * 100), 2).sort_values(ascending=False)

    missing_df = pd.concat([total, pct], axis=1, keys=["Total", "Percent"]).reset_index()

    print(missing_df)

calc_missing(train_data)

#%%
# KNN Imputer
knn_train_copy = train_data.select_dtypes(include="number").copy(deep = True)
knn_imputer = KNNImputer(n_neighbors = 2, weights = "uniform")
knn_train_imputed = knn_imputer.fit_transform(knn_train_copy)

# calc_missing(knn_train_imputed)


#%%
# sns.scatterplot(x = train_data["longitude"], y = train_data["latitude"])
# train_data.plot(kind = "scatter", x = "longitude", y = "latitude", 
# alpha = 0.4, c = "price", cmap=plt.get_cmap("jet"), colorbar=True)

#%%
# Missing values in host
train_data.loc[:, list(missing_data.iloc[:4, 0])]

#%%
# Fill missing values for host_response_rate and host_acceptance_rate
train_data["host_response_rate"] = train_data["host_response_rate"].str.split('%').str[0].fillna('0').astype("int")
train_data["host_acceptance_rate"] = train_data["host_acceptance_rate"].str.split('%').str[0].fillna('0').astype("int")

#%%
# Missing values in reviews
train_data.loc[:, list(missing_data.iloc[6:16, 0])]

#%%
# Fill missing values for review_scores
review_cols = [col for col in train_data.columns if "review_scores" in col]
review_cols.append("reviews_per_month")

train_data[review_cols] = train_data[review_cols].fillna(0)

#%%
# Missing values in neighborhood
train_data.loc[:, list(missing_data.iloc[4:6, 0])]
train_data["host_neighbourhood"]

#%%
# See columns that contain neighbo(u)r
train_data.filter(regex="neighb")

#%%
# Drop uncleaned columns
train_data.drop(["host_neighbourhood", "neighbourhood"], axis=1, inplace=True)

#%%
# Missing values in bedroom
train_data["bedrooms"]

print(calc_missing(train_data))
