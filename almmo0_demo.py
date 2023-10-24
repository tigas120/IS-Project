# %% Import packages
from almmo0 import ALMMo0
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, cohen_kappa_score
import numpy as np

# Load dataset
data  = pd.read_csv("final_music_dataset.csv")

# Split the data into features (X) and target (y)
X = data.drop(['Class;;','Artist Name','Track Name'], axis=1)
Y = data['Class;;']

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

#create matrix for evaluation of performance in regards to the number of used parameters
evaluation = [[0 for i in range(X.shape[1])] for i in range(6)]
evaluation[0][0] = "numb_feat"
evaluation[1][0] = "accuracy"
evaluation[2][0] = "recall"
evaluation[3][0] = "precision"
evaluation[4][0] = "kappa"
evaluation[5][0] = "f1"

#train a model for each number of attributes
for i in range(1,X.shape[1]):
    k_best = SelectKBest(score_func=f_classif,k=i)
    X_new = k_best.fit_transform(X, Y)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.2)

    model = ALMMo0()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluation[0][i] = i
    acc_score = accuracy_score(y_test, y_pred)
    evaluation[1][i] = acc_score
    cm = confusion_matrix(y_test, y_pred)
    recall = np.nanmean(np.diag(cm) / np.sum(cm, axis = 1))
    evaluation[2][i] = recall
    precision = np.nanmean(np.diag(cm) / np.sum(cm, axis = 0))
    evaluation[3][i] = precision
    kappa = cohen_kappa_score(y_test, y_pred)
    evaluation[4][i] = kappa
    f1 = 2 * (precision * recall) / (precision + recall)
    evaluation[5][i] = f1



    print(i)

#evaluate which amount of parameters is the best one
print(*evaluation, sep='\n')

    

