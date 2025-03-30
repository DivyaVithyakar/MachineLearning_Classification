import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

# Load dataset
dataset = pd.read_csv("../data/Social_Network_Ads.csv")

# Convert categorical variables into dummy/one-hot encoding
dataset = pd.get_dummies(dataset, drop_first=True)

# Define independent and dependent variables
independent = dataset[['User ID', 'Age', 'EstimatedSalary', 'Gender_Male']]
dependent = dataset[['Purchased']]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)

# Convert numerical data into categorical form (e.g., binning)
x_train = x_train.apply(lambda col: pd.cut(col, bins=10, labels=False))  # Convert to categorical bins
x_test = x_test.apply(lambda col: pd.cut(col, bins=10, labels=False))  # Convert to categorical bins

# Define parameter grid for GridSearchCV
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],  # Smoothing parameter
    'fit_prior': [True, False]  # Whether to learn class priors
}

# Initialize GridSearchCV
grid = GridSearchCV(estimator=CategoricalNB(), param_grid=param_grid, cv=5, refit=True, scoring='accuracy', n_jobs=-1, verbose=1)

# Train the model
grid.fit(x_train, y_train.values.ravel())

# Print best hyperparameters
print("Best Parameters:", grid.best_params_)

# Make predictions
y_predict = grid.predict(x_test)

# Evaluate performance
matrix = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:\n", matrix)

clf_report = classification_report(y_test, y_predict)
print("Classification Report:\n", clf_report)

acc_score = accuracy_score(y_test, y_predict)
print("Accuracy Score:", acc_score)

# ROC-AUC Score
roc_score = roc_auc_score(y_test, grid.predict_proba(x_test)[:, 1])
print("ROC-AUC Score:", roc_score)
