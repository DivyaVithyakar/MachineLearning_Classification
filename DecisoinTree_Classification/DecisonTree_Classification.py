import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report


dataset = pd.read_csv("../data/Social_Network_Ads.csv")
dataset = pd.get_dummies(dataset, drop_first=True)
print(dataset.columns)
independent = dataset[['User ID', 'Gender_Male', 'Age', 'EstimatedSalary']]
dependent = dataset[['Purchased']]
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)
criterions = ['gini', 'entropy', 'log_loss']
splitters = ['best','random']

best_accuracy = 0
best_model = None
best_params = {}


for criterion in criterions:
    for splitter in splitters:
        try:
            classifier = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
            classifier.fit(x_train,y_train)
            y_predict = classifier.predict(x_test)
            matrix = confusion_matrix(y_test, y_predict)
            print(matrix)
            clf_report = classification_report(y_test, y_predict)
            print(clf_report)
            acc_score = accuracy_score(y_test, y_predict)
            print(acc_score)

            if acc_score > best_accuracy:
                best_accuracy = acc_score
                best_model = classifier
                best_params = {'criterion' : criterion, 'splitter' : splitter}

        except Exception as e:
            print(f'Criterion: {criterion}, splitter: {splitter}, Error: {str(e)}')
