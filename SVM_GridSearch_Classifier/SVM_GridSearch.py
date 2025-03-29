import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,roc_auc_score

# Load dataset
dataset = pd.read_csv("../data/Social_Network_Ads.csv")
dataset = pd.get_dummies(dataset, drop_first=True)

# Splitting independent and dependent variables
independent = dataset[['User ID', 'Age', 'EstimatedSalary', 'Gender_Male']]
dependent = dataset[['Purchased']]
x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)

# Define hyperparameter grid
param_grid = {
    'C': [10,100,1000,2000,3000],  # Regularization parameter
    'kernel': ['linear', 'poly'],  # Kernel type
    'gamma': ['scale', 'auto'],  # Kernel coefficient
}

# Perform GridSearchCV
grid = GridSearchCV(estimator=SVC(probability=True), param_grid=param_grid, cv=3, refit=True, scoring='f1_weighted', n_jobs=-1, verbose=1)
grid.fit(x_train, y_train.values.ravel())  # Convert y_train to 1D array

# Print best parameters
print("Best Parameters:", grid.best_params_)

# Get best model
#best_model = grid.best_estimator_

# Make predictions
y_predict = grid.predict(x_test)

# Evaluation metrics
matrix = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:\n", matrix)

clf_report = classification_report(y_test, y_predict)
print("Classification Report:\n", clf_report)

acc_score = accuracy_score(y_test, y_predict)
print("Accuracy Score:", acc_score)

roc_score = roc_auc_score(y_test, grid.predict_proba(x_test)[:, 1])
print(roc_score)
