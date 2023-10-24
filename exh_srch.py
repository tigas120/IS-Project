import numpy as np
from itertools import combinations
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#%%
# Load your dataset from a CSV file
data = pd.read_csv("final_music_dataset.csv", on_bad_lines='skip')

# Drop rows with missing values
data.dropna(inplace=True)

# Separate X and y
X = data.loc[:, ["Popularity", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_in min/ms", "time_signature"]]

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["Class;;"])
#%%
def exhaustive_feature_selection(X, y, model, n_features_to_select):
    n_features = X.shape[1]
    best_feature_set = None
    best_score = -np.inf

    for feature_set in combinations(X.columns, n_features_to_select):
        selected_features = X[list(feature_set)]
        scores = cross_val_score(model, selected_features, y, cv=5, scoring='accuracy')
        avg_score = np.mean(scores)

        if avg_score > best_score:
            best_score = avg_score
            best_feature_set = feature_set

    return best_feature_set, best_score
#%%
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', random_state=42)
#%%
n_features_to_select = 10  # Set the number of features you want to select
best_features, best_score = exhaustive_feature_selection(X, y, mlp, n_features_to_select)

print("Best Feature Set:", best_features)
print("Best Cross-Validation Score:", best_score)
