import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#investigate amounts of songs in each class after cleanup of the dataset
for i in range(11):
    data  = pd.read_csv("final_music_dataset.csv")
    data = data.drop(data[data['Class;;'] != i].index)
    print("Songs in class ",i,": ")
    print(data.shape[0])

#plot classes over features
data  = pd.read_csv("final_music_dataset.csv")
X = data.loc[:,["Popularity", "danceability", "energy", "key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_in min/ms","time_signature","Class;;"]]
y = data["Class;;"]
plot = sns.pairplot(X, hue = 'Class;;',palette='tab10')
plot.savefig("plot_features.png")

#plot pearson correlation of features
X = data.loc[:,["Popularity", "danceability", "energy", "key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_in min/ms","time_signature"]]
y = data["Class;;"]
plt.figure(figsize=(12,10))
# pearson correlation
cor = X.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.savefig("correlation_pearson.png")