import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

dataset = pd.read_csv("../data/Social_Network_Ads.csv")
dataset = pd.get_dummies(dataset, drop_first=True)
#print(dataset.columns)
independent = dataset[['User ID', 'Age', 'EstimatedSalary', 'Gender_Male']]
dependent = dataset[['Purchased']]
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)

param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],  # Different splitting criteria
    'splitter': ['best', 'random'],  # How nodes are split
    'max_depth': [None, 10, 20, 30],  # Controls tree depth to prevent overfitting
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required in a leaf node
    'max_features': [None, 'sqrt', 'log2'],  # Number of features to consider for the best split
    'max_leaf_nodes': [None, 10, 20, 50],  # Maximum leaf nodes in the tree
    'min_impurity_decrease': [0.0, 0.01, 0.1],  # Threshold for splitting nodes
}

grid = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1,verbose=1)
grid.fit(x_train,y_train)
print(grid.best_params_)
best_model = grid.best_estimator_
y_predict = best_model.predict(x_test)

matrix = confusion_matrix(y_test, y_predict)
print(matrix)
clf_report = classification_report(y_test, y_predict)
print(clf_report)
acc_score = accuracy_score(y_test, y_predict)
print(acc_score)

