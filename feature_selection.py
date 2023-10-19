import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# %%
data  = pd.read_csv("final_music_dataset.csv",  on_bad_lines='skip') 
data.head()
# dropping null value columns to avoid errors
data.dropna(inplace = True)

#%% Separate X and Y
X = data.loc[:,["Popularity", "danceability", "energy", "key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_in min/ms","time_signature"]]
Y = data["Class;;"]

#%% Apply the feature selection
#   Change the value of 'k=' to the desired number
k_best = SelectKBest(score_func=f_classif,k='all')
X_new = k_best.fit_transform(X, Y)

#%% Print the score of each feature
scores = k_best.scores_

for i, score in enumerate(scores):
    print(f"Feature {i}: {score}")
