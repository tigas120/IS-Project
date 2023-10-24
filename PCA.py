import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, cohen_kappa_score
# %%
data  = pd.read_csv("final_music_dataset.csv",  on_bad_lines='skip') 
data.head()
# dropping null value columns to avoid errors
data.dropna(inplace = True)

#%% Separate X and Y
X = data.loc[:,["Popularity", "danceability", "energy", "key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_in min/ms","time_signature"]]
y = data["Class;;"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%

# Applying PCA function on training
# and testing set of X component
 
pca = PCA(n_components = 10)
components = pca.fit_transform(X)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
 
explained_variance = pca.explained_variance_ratio_

# %% Train model
regr = MLPClassifier(hidden_layer_sizes=(90,40),random_state=42, max_iter=500)
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

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

for i in range(10):
    plt.subplot(3, 4, i + 1)  # Adjust the subplot layout as needed
    plt.scatter(components[:, i], np.zeros_like(components[:, i]))
    plt.title(f'Principal Component {i+1}')

plt.tight_layout()
plt.show()