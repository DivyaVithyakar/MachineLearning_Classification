import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report


dataset = pd.read_csv("../data/Social_Network_Ads.csv")
dataset = pd.get_dummies(dataset, drop_first=True)
print(dataset.columns)
independent = dataset[['Gender_Male', 'Age', 'EstimatedSalary']]
dependent = dataset[['Purchased']]
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

C_values = [0.1, 1, 10]  # Regularization parameter
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
gammas = ['scale', 'auto', 0.01, 0.1, 1]  # Kernel coefficient
degrees = [2, 3, 4]  # Only relevant for 'poly' kernel

# Store Best Model
best_accuracy = 0
best_model = None
best_params = {}

for C in C_values:
    for kernel in kernels:
        for gamma in gammas:
            for degree in degrees if kernel == 'poly' else [3]:  # Only test degree for 'poly'
                try:
                    # Train SVM Model
                    classifier = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree if kernel == 'poly' else 3)
                    classifier.fit(x_train, y_train)
                    y_predict = classifier.predict(x_test)

                    # Evaluate Performance
                    matrix = confusion_matrix(y_test, y_predict)
                    print(matrix)
                    clf_report = classification_report(y_test, y_predict)
                    print(clf_report)
                    acc_score = accuracy_score(y_test, y_predict)
                    print(acc_score)

                    if acc_score > best_accuracy:
                        best_accuracy = acc_score
                        best_model = classifier
                        best_params = {'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree}
                except Exception as e:
                    print(f"C: {C}, Kernel: {kernel}, Gamma: {gamma}, Degree: {degree}, Error: {str(e)}")
