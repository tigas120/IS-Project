import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score
from sklearn import preprocessing

data  = pd.read_csv("final_music_dataset.csv")

# Split the data into features (X) and target (y)
X = data.drop(['Class;;','Artist Name','Track Name'], axis=1)
y = data['Class;;']

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(acc_score))
kappa = cohen_kappa_score(y_test, y_pred)
print("Kappa Score: {:.3f}".format(kappa))

cm = confusion_matrix(y_test, y_pred)
recall = np.nanmean(np.diag(cm) / np.sum(cm, axis = 1))
precision = np.nanmean(np.diag(cm) / np.sum(cm, axis = 0))

print("Recall: {:.3f}".format(recall))
print("Precision: {:.3f}".format(precision))

f1 = 2 * (precision * recall) / (precision + recall)
print("F1: ",f1)