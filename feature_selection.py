import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# %%
data  = pd.read_csv("music_dataset.csv",  on_bad_lines='skip') 
data.head()
# dropping null value columns to avoid errors
data.dropna(inplace = True)

# Define a custom filtering condition to identify lines with more than 16 commas
condition = data.apply(lambda row: row.str.count(',').sum() <= 16, axis=1)

# Create a new DataFrame without lines that don't meet the condition
clean_data = data[condition]

# Assuming "column_name" is the name of the 3rd column
filtered_data = clean_data[pd.to_numeric(clean_data['Popularity'], errors='coerce').notna()]

# Remove ';;' and convert to string
filtered_data['time_signature'] = filtered_data['time_signature'].str.replace(';','').astype(float)

# Clean 'duration_in min/ms' column using regular expression
filtered_data['duration_in min/ms'] = filtered_data['duration_in min/ms'].apply(lambda x: float(re.sub(r'[^\d.]', '', x)) if isinstance(x, str) else x)

#%%
X = filtered_data.loc[:,["Popularity", "danceability", "energy", "key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_in min/ms","time_signature"]]
Y = filtered_data["Class;;"]

#%%

k_best = SelectKBest(score_func=f_classif,k='all')
X_new = k_best.fit_transform(X, Y)

#%%
scores = k_best.scores_

for i, score in enumerate(scores):
    print(f"Feature {i}: {score}")
