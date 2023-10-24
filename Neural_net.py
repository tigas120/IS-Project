# %% Imports
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
from numpy import savetxt
from sklearn.metrics import confusion_matrix
import numpy as np

# %% Load dataset and create train-test sets
# %%
data  = pd.read_csv("final_music_dataset.csv",  on_bad_lines='skip') 
data.head()
# dropping null value columns to avoid errors
data.dropna(inplace = True)

#%% Separate X and Y
X = data.loc[:,["Popularity", "danceability", "energy", "key","loudness","mode","speechiness","acousticness","instrumentalness","valence"]]
y = data["Class;;"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% Train model
regr = MLPClassifier(hidden_layer_sizes=(90,50  ),random_state=42, max_iter=400,activation='relu',alpha=0.001,learning_rate='constant')
regr.fit(X_train, y_train)

# %% Get model predictions
y_pred = regr.predict(X_test)
# %% Compute classification metrics
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(acc_score))
kappa = cohen_kappa_score(y_test, y_pred)
print("Kappa Score: {:.3f}".format(kappa))

cm = confusion_matrix(y_test, y_pred)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)
f1 = 2 * (precision * recall) / (precision + recall)
#%%
print("F1 Score: {:.3f} ".format(np.nanmean(f1)))
print("Recall: {:.3f}".format(np.nanmean(recall)))
print("Precision: {:.3f}".format(np.nanmean(precision)))