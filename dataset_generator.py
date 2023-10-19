import pandas as pd
import re

#%%
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

# Clean all the parameters to save dataset
filtered_data['time_signature'] = filtered_data['time_signature'].str.replace(';','').astype(float)
filtered_data['duration_in min/ms'] = filtered_data['duration_in min/ms'].apply(lambda x: float(re.sub(r'[^\d.]', '', x)) if isinstance(x, str) else x)
filtered_data['Class;;'] = filtered_data['Class;;'].str.replace(';;','')
filtered_data['Class;;'] = filtered_data['Class;;'].str.replace(';','')
filtered_data['Class;;'] = filtered_data['Class;;'].astype(float)

#%% Save the post-processed dataset to use in the models
filtered_data.to_csv('final_music_dataset.csv',index=True)