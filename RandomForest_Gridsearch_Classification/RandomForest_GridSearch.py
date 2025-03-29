import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import  pickle

dataset = pd.read_csv("../data/Social_Network_Ads.csv")
dataset = pd.get_dummies(dataset, drop_first=True)
#print(dataset.columns)
independent = dataset[['User ID', 'Age', 'EstimatedSalary', 'Gender_Male']]
dependent = dataset[['Purchased']]
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)
param_grid = {
    'n_estimators': [10, 50, 100],  # Corrected parameter name
    'criterion': ['log_loss', 'entropy', 'gini'],  # Corrected parameter name
    'max_depth': [None, 10, 20],  # Optional: Limit depth to prevent overfitting
    'min_samples_split': [2, 5, 10],  # Control tree complexity
    'min_samples_leaf': [1, 2, 4]  # Control minimum leaf nodes
}
grid = GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(x_train,y_train.values.ravel())
print(grid.best_params_)
best_model = grid.best_estimator_
print(best_model)
y_predict = best_model.predict(x_test)
matrix = confusion_matrix(y_test,y_predict)
print(matrix)
clf_report = classification_report(y_test, y_predict)
print(clf_report)
acc_score = accuracy_score(y_test, y_predict)
print(acc_score)

filename = "../data/finalised_randomForestGridSearch_model.sav"
pickle.dump(best_model, open(filename, 'wb'))