import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from scipy.stats import randint
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
#load data
data  = pd.read_csv("final_music_dataset.csv")

# Split the data into features (X) and target (y)
X = data.drop(['Class;;','Artist Name','Track Name'], axis=1)
y = data['Class;;']

#normalize data
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

#create matrix for evaluation of performance in regards to the number of used parameters
evaluation = [[0 for i in range(X.shape[1])] for i in range(4)]
evaluation[0][0] = "numb_feat"
evaluation[1][0] = "accuracy"
evaluation[2][0] = "recall"
evaluation[3][0] = "precision"

#loop over the amount of used parameters
for i in range(1,X.shape[1]):
    k_best = SelectKBest(score_func=f_classif,k=i)
    X_new = k_best.fit_transform(X, y)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)

    knn = KNeighborsClassifier()
    #create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 25)}
    #use gridsearch to test all values for n_neighbors
    knn_gs = GridSearchCV(knn, params_knn, cv=5)
    #fit model to training data
    knn_gs.fit(X_train, y_train)

    y_pred = knn_gs.predict(X_test)

    evaluation[0][i] = i

    acc_score = accuracy_score(y_test, y_pred)
    evaluation[1][i] = acc_score


    cm = confusion_matrix(y_test, y_pred)
    recall = np.mean(np.diag(cm) / np.sum(cm, axis = 1))
    evaluation[2][i] = recall
    precision = np.mean(np.diag(cm) / np.sum(cm, axis = 0))
    evaluation[3][i] = precision
    print(i)


#evaluate which amount of parameters is the best one
print(*evaluation, sep='\n')


