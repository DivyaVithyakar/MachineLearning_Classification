import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,roc_auc_score
from sklearn.metrics import classification_report


dataset = pd.read_csv("../data/Social_Network_Ads.csv")
dataset = pd.get_dummies(dataset, drop_first=True)
#print(dataset.columns)
independent = dataset[['User ID', 'Age', 'EstimatedSalary', 'Gender_Male']]
dependent = dataset[['Purchased']]
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)

# Define hyperparameter grid for KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors
    'weights': ['uniform', 'distance'],  # Weighting scheme
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metrics
}

# Apply GridSearchCV
grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5, refit=True, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(x_train, y_train.values.ravel())

# Print best parameters
print("Best Parameters:", grid.best_params_)

# Predict on test data
y_pred = grid.predict(x_test)

# Evaluation Metrics
matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", matrix)

clf_report = classification_report(y_test, y_pred)
print("Classification Report:\n", clf_report)

acc_score = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {acc_score:.4f}")

# ROC-AUC Score
roc_score = roc_auc_score(y_test, grid.predict_proba(x_test)[:, 1])
print(f"ROC AUC Score: {roc_score:.4f}")