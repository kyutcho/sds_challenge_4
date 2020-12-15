import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

train_data = pd.read_csv("data/public_listings.csv")

# Columns with mixed type
train_data.columns[[15, 24, 33, 38, 42, 43, 51]]

train_data[train_data.columns[[15, 24, 33, 38, 42, 43, 51]]]

# Row 17613 causes a problem - Read csv file again
train_data = pd.read_csv("data/public_listings.csv", skiprows=[17614])

# Check the first 3 rows
train_data.head(3)

# Shape
train_data.shape

# Basic info
train_data.info()

# Missing values
total = train_data.isnull().sum().sort_values(ascending=False)
pct = round((train_data.isnull().sum() / len(train_data) * 100), 2).sort_values(ascending=False)

missing_data = pd.concat([total, pct], axis=1, keys=["Total", "Percent"])
missing_data.head(10)

missing_data.head(4)
