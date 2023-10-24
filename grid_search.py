import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# Load and preprocess your dataset
data = pd.read_csv("final_music_dataset.csv", on_bad_lines='skip')
data.dropna(inplace=True)

# Define feature matrix X and target variable y
X = data.loc[:, ["Popularity", "danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_in min/ms", "time_signature"]]
y = data["Class;;"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(90, 50), (90, 40)],  # Different architectures
    'activation': ['relu', 'logistic'],  # Activation functions
    'alpha': [0.0001, 0.001],  # Regularization strength
    'learning_rate': ['constant', 'invscaling'],  # Learning rate schedule
    'max_iter': [400, 450]  # Maximum training iterations
}

# Create the MLP classifier
mlp = MLPClassifier(random_state=42)

# Create the GridSearchCV object
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Perform Grid Search
grid_search.fit(X, y)

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# You can access and print the best hyperparameters as follows:
#print("Best Hyperparameters:", best_params)
# =============================================================================
# # Create the GridSearchCV object
# grid_search = GridSearchCV(MLPClassifier(random_state=42, max_iter=500), param_grid, cv=10, scoring='accuracy')
# 
# # Perform Grid Search
# grid_search.fit(X_train, y_train)
# 
# # Get the best parameters and best model
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_
# =============================================================================
#%%
# Evaluate the Best Model
best_model.fit(X_train, y_train)
accuracy = best_model.score(X_test, y_test)

print("Best Hyperparameters:", best_params)
print("Test Accuracy of the Best Model:", accuracy)
