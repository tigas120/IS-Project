# %% Imports
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from pyfume.Clustering import Clusterer
from pyfume.EstimateAntecendentSet import AntecedentEstimator
from pyfume.EstimateConsequentParameters import ConsequentEstimator
from pyfume.SimpfulModelBuilder import SugenoFISBuilder
from pyfume.Tester import SugenoFISTester
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score
from numpy import clip, column_stack, argmax, vectorize
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import numpy as np
# %% Load dataset and create train-test sets
import pandas as pd
import re

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
y = filtered_data["Class;;"]

#%%

#k_best = SelectKBest(score_func=f_classif,k='all')
k_best = SelectKBest(score_func=f_classif,k='all')
X = k_best.fit_transform(X, y)

#%%
scores = k_best.scores_

#for i, score in enumerate(scores):
 #   print(f"Feature {i}: {score}")



min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=92)

#var_names = ["Popularity", "danceability", "energy", "key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_in min/ms","time_signature"]
#var_names = [var_names[i].title().replace(' ','') for i in range(0, len(var_names))]
#var_names = [var_names[i].title().replace('/','') for i in range(0, len(var_names))]

num_classes = 11
num_feat = X.shape[1]
print("Number of chosen attributes: ",num_feat )

var_names = ["attribute" for i in  range(num_feat)]

# Create 0-vs-all target vector
#y_train_0_vs_all = vectorize({0:1,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0}.get)(y_train)

#train 0 vs all model
y_train_0_vs_all = list(y_train)
for f in range(len(y_train)):
        if y_train_0_vs_all[f] == 0:
            y_train_0_vs_all[f] = 100
        else:
            y_train_0_vs_all[f] = 0

for f in range(len(y_train)):
    if y_train_0_vs_all[f] == 100:
        y_train_0_vs_all[f] = 1
# Cluster the input-output space
cl = Clusterer(x_train=X_train, y_train=y_train_0_vs_all, nr_clus=20)
clust_centers, part_matrix, _ = cl.cluster(method='fcm')
# Estimate membership functions parameters
ae = AntecedentEstimator(X_train, part_matrix)
antecedent_params = ae.determineMF()
# Estimate consequent parameters
ce = ConsequentEstimator(X_train, y_train_0_vs_all, part_matrix)
conseq_params = ce.suglms()
# Build first-order Takagi-Sugeno model
modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
model_0_vs_all = modbuilder.get_model()


modtester = SugenoFISTester(model_0_vs_all, X_test, var_names)
y_pred_probs_0_vs_all = clip(modtester.predict()[0], 0, 1)
y_pred_probs_0_vs_all = column_stack((1 - y_pred_probs_0_vs_all, y_pred_probs_0_vs_all))
print("Y-Pred", y_pred_probs_0_vs_all)










"""
y_train_i_vs_all = {}
model_i_vs_all = {}
y_pred_probs_i_vs_all = {}

# Create i-vs-all target vector
for i in range(num_classes):
    y_train_i_vs_all[i] = list(y_train)
    for f in range(len(y_train)):
        if y_train_i_vs_all[i][f] == i:
            y_train_i_vs_all[i][f] = 100
        else:
            y_train_i_vs_all[i][f] = 0

    for f in range(len(y_train)):
        if y_train_i_vs_all[i][f] == 100:
            y_train_i_vs_all[i][f] = 1

    # Cluster the input-output space
    cl = Clusterer(x_train=X_train, y_train=y_train_i_vs_all[i], nr_clus=9)
    clust_centers, part_matrix, _ = cl.cluster(method='fcm')
    # Estimate membership functions parameters
    ae = AntecedentEstimator(X_train, part_matrix)
    antecedent_params = ae.determineMF()
    # Estimate consequent parameters
    ce = ConsequentEstimator(X_train, y_train_i_vs_all[i], part_matrix)
    conseq_params = ce.suglms()
    # Build first-order Takagi-Sugeno model
    modbuilder = SugenoFISBuilder(antecedent_params, conseq_params, var_names, save_simpful_code=False)
   # model_0_vs_all = modbuilder.get_model()
    model_i_vs_all[i] = modbuilder.get_model()

    modtester = SugenoFISTester(model_i_vs_all[i], X_test, var_names)
    y_pred_probs_i_vs_all[i] = clip(modtester.predict()[0], 0, 1)
    y_pred_probs_i_vs_all[i] = column_stack((1 - y_pred_probs_i_vs_all[i], y_pred_probs_i_vs_all[i]))
    print("Y-Pred", y_pred_probs_i_vs_all[i])
    #modell sagt dass keins zu der Klasse gehört, und zwar immer


# %% Aggregate class probabilities and get class predictions

y_pred_probs_final = column_stack((y_pred_probs_i_vs_all[0][:,1],y_pred_probs_i_vs_all[0][:,0],y_pred_probs_i_vs_all[0][:,0],y_pred_probs_i_vs_all[0][:,0],y_pred_probs_i_vs_all[0][:,0],y_pred_probs_i_vs_all[0][:,0],y_pred_probs_i_vs_all[0][:,0],y_pred_probs_i_vs_all[0][:,0],y_pred_probs_i_vs_all[0][:,0],y_pred_probs_i_vs_all[0][:,0],y_pred_probs_i_vs_all[0][:,0])) +\
        column_stack((y_pred_probs_i_vs_all[1][:,0],y_pred_probs_i_vs_all[1][:,1],y_pred_probs_i_vs_all[1][:,0],y_pred_probs_i_vs_all[1][:,0],y_pred_probs_i_vs_all[1][:,0],y_pred_probs_i_vs_all[1][:,0],y_pred_probs_i_vs_all[1][:,0],y_pred_probs_i_vs_all[1][:,0],y_pred_probs_i_vs_all[1][:,0],y_pred_probs_i_vs_all[1][:,0],y_pred_probs_i_vs_all[1][:,0])) +\
        column_stack((y_pred_probs_i_vs_all[2][:,0],y_pred_probs_i_vs_all[2][:,0],y_pred_probs_i_vs_all[2][:,1],y_pred_probs_i_vs_all[2][:,0],y_pred_probs_i_vs_all[2][:,0],y_pred_probs_i_vs_all[2][:,0],y_pred_probs_i_vs_all[2][:,0],y_pred_probs_i_vs_all[2][:,0],y_pred_probs_i_vs_all[2][:,0],y_pred_probs_i_vs_all[2][:,0],y_pred_probs_i_vs_all[2][:,0])) +\
        column_stack((y_pred_probs_i_vs_all[3][:,0],y_pred_probs_i_vs_all[3][:,0],y_pred_probs_i_vs_all[3][:,0],y_pred_probs_i_vs_all[3][:,1],y_pred_probs_i_vs_all[3][:,0],y_pred_probs_i_vs_all[3][:,0],y_pred_probs_i_vs_all[3][:,0],y_pred_probs_i_vs_all[3][:,0],y_pred_probs_i_vs_all[3][:,0],y_pred_probs_i_vs_all[3][:,0],y_pred_probs_i_vs_all[3][:,0])) +\
        column_stack((y_pred_probs_i_vs_all[4][:,0],y_pred_probs_i_vs_all[4][:,0],y_pred_probs_i_vs_all[4][:,0],y_pred_probs_i_vs_all[4][:,0],y_pred_probs_i_vs_all[4][:,1],y_pred_probs_i_vs_all[4][:,0],y_pred_probs_i_vs_all[4][:,0],y_pred_probs_i_vs_all[4][:,0],y_pred_probs_i_vs_all[4][:,0],y_pred_probs_i_vs_all[4][:,0],y_pred_probs_i_vs_all[4][:,0])) +\
        column_stack((y_pred_probs_i_vs_all[5][:,0],y_pred_probs_i_vs_all[5][:,0],y_pred_probs_i_vs_all[5][:,0],y_pred_probs_i_vs_all[5][:,0],y_pred_probs_i_vs_all[5][:,0],y_pred_probs_i_vs_all[5][:,1],y_pred_probs_i_vs_all[5][:,0],y_pred_probs_i_vs_all[5][:,0],y_pred_probs_i_vs_all[5][:,0],y_pred_probs_i_vs_all[5][:,0],y_pred_probs_i_vs_all[5][:,0])) +\
        column_stack((y_pred_probs_i_vs_all[6][:,0],y_pred_probs_i_vs_all[6][:,0],y_pred_probs_i_vs_all[6][:,0],y_pred_probs_i_vs_all[6][:,0],y_pred_probs_i_vs_all[6][:,0],y_pred_probs_i_vs_all[6][:,0],y_pred_probs_i_vs_all[6][:,1],y_pred_probs_i_vs_all[6][:,0],y_pred_probs_i_vs_all[6][:,0],y_pred_probs_i_vs_all[6][:,0],y_pred_probs_i_vs_all[6][:,0])) +\
        column_stack((y_pred_probs_i_vs_all[7][:,0],y_pred_probs_i_vs_all[7][:,0],y_pred_probs_i_vs_all[7][:,0],y_pred_probs_i_vs_all[7][:,0],y_pred_probs_i_vs_all[7][:,0],y_pred_probs_i_vs_all[7][:,0],y_pred_probs_i_vs_all[7][:,0],y_pred_probs_i_vs_all[7][:,1],y_pred_probs_i_vs_all[7][:,0],y_pred_probs_i_vs_all[7][:,0],y_pred_probs_i_vs_all[7][:,0])) +\
        column_stack((y_pred_probs_i_vs_all[8][:,0],y_pred_probs_i_vs_all[8][:,0],y_pred_probs_i_vs_all[8][:,0],y_pred_probs_i_vs_all[8][:,0],y_pred_probs_i_vs_all[8][:,0],y_pred_probs_i_vs_all[8][:,0],y_pred_probs_i_vs_all[8][:,0],y_pred_probs_i_vs_all[8][:,0],y_pred_probs_i_vs_all[8][:,1],y_pred_probs_i_vs_all[8][:,0],y_pred_probs_i_vs_all[8][:,0])) +\
        column_stack((y_pred_probs_i_vs_all[9][:,0],y_pred_probs_i_vs_all[9][:,0],y_pred_probs_i_vs_all[9][:,0],y_pred_probs_i_vs_all[9][:,0],y_pred_probs_i_vs_all[9][:,0],y_pred_probs_i_vs_all[9][:,0],y_pred_probs_i_vs_all[9][:,0],y_pred_probs_i_vs_all[9][:,0],y_pred_probs_i_vs_all[9][:,0],y_pred_probs_i_vs_all[9][:,1],y_pred_probs_i_vs_all[9][:,0])) +\
        column_stack((y_pred_probs_i_vs_all[10][:,0],y_pred_probs_i_vs_all[10][:,0],y_pred_probs_i_vs_all[10][:,0],y_pred_probs_i_vs_all[10][:,0],y_pred_probs_i_vs_all[10][:,0],y_pred_probs_i_vs_all[10][:,0],y_pred_probs_i_vs_all[10][:,0],y_pred_probs_i_vs_all[10][:,0],y_pred_probs_i_vs_all[10][:,0],y_pred_probs_i_vs_all[10][:,0],y_pred_probs_i_vs_all[10][:,1]))


print("column stack", y_pred_probs_final)
y_pred_probs_final = y_pred_probs_final/y_pred_probs_final.sum(axis=1,keepdims=1)
print("runtergebrochen", y_pred_probs_final)
y_pred = argmax(y_pred_probs_final,axis=1)

print("y_test: ", y_test)
print("y_pred: ", y_pred)



# %% Compute classification metrics
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3f}".format(acc_score))
kappa = cohen_kappa_score(y_test, y_pred)
print("Kappa Score: {:.3f}".format(kappa))

cm = confusion_matrix(y_test, y_pred)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)

print("Recall: {:.3f}".format(np.mean(recall)))
print("Precision: {:.3f}".format(np.mean(precision)))

"""