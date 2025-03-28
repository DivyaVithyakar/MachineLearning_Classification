import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

dataset = pd.read_csv("../data/Social_Network_Ads.csv")
dataset = pd.get_dummies(dataset, drop_first=True)
independent = dataset[['User ID', 'Gender_Male', 'Age', 'EstimatedSalary']]
dependent = dataset['Purchased']
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)

n_estimator = [10,50,100]
criterions = ['log_loss', 'entropy', 'gini']

best_accuracy = 0
best_model = None
best_params = {}

for criterion in criterions:
    for n_estimators in n_estimator:
        try:
            # Train the model
            classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
            classifier.fit(x_train,y_train.values.ravel())
            y_predict = classifier.predict(x_test)
            matrix = confusion_matrix(y_test, y_predict)
            print(matrix)
            clf_report = classification_report(y_test, y_predict)
            print(clf_report)
            acc_score = accuracy_score(y_test, y_predict)
            print(f'Criterion: {criterion}, n_estimators: {n_estimators}, accuracy_score: {acc_score}')

            # Update the best model if the current one is better
            if acc_score > best_accuracy:
                best_accuracy = acc_score
                best_model = classifier
                best_params = {'criterion': criterion, 'n_estimators': n_estimators}

        except Exception as e:
            print(f'Criterion: {criterion}, n_estimators: {n_estimators}, Error: {str(e)}')
