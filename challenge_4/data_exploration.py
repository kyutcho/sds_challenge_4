import numpy as npimport pandas as pdimport matplotlib.pyplot as pltimport seaborn as snstrain_data = pd.read_csv("data/public_listings.csv")# Columns with mixed typetrain_data.columns[[15,24,33,38,42,43,51]]train_data.head()train_data.shape