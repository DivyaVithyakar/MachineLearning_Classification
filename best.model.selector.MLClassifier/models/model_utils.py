import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Load dataset
def load_data(file_path, reference_columns=None):
    dataset = pd.read_csv(file_path)
    dataset = pd.get_dummies(dataset, drop_first=True)

    if reference_columns is not None:
        dataset = dataset.reindex(columns=reference_columns, fill_value=0)
    return dataset

#Preprocess Data
def preprocess_data(dataset, target_column):
    independent = dataset.drop(columns=[target_column])
    dependent = dataset[target_column]

    X_train, X_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Preprocessing functions
def preprocess_minmax(X_train, X_test):
    """Applies MinMax Scaling to ensure non-negative values."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def preprocess_binarizer(X_train, X_test):
    """Binarizes data (for BernoulliNB)."""
    binarizer = Binarizer(threshold=0.0)
    return binarizer.fit_transform(X_train), binarizer.transform(X_test)

def preprocess_default(X_train, X_test):
    """Returns data as-is (for models that can handle negative values)."""
    return X_train, X_test


# Model functions
def get_logistic_regression():
    return LogisticRegression(), {'C': [0.1, 1, 10], 'solver': ['liblinear']}, preprocess_default

def get_knn():
    return KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}, preprocess_default

def get_gaussian_nb():
    return GaussianNB(), {}, preprocess_default

def get_multinomial_nb():
    return MultinomialNB(), {'alpha': [0.1, 0.5, 1.0]}, preprocess_minmax  # Needs non-negative values

def get_bernoulli_nb():
    return BernoulliNB(), {'alpha': [0.1, 0.5, 1.0]}, preprocess_binarizer  # Needs binary values

def get_complement_nb():
    return ComplementNB(), {'alpha': [0.1, 0.5, 1.0]}, preprocess_minmax  # Needs non-negative values

def get_categorical_nb():
    return CategoricalNB(), {'alpha': [0.1, 0.5, 1.0]}, preprocess_default  # Assumes categorical encoding

def get_svm():
    return SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}, preprocess_default

def get_decision_tree():
    return DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20]}, preprocess_default

def get_random_forest():
    return RandomForestClassifier(), {'n_estimators': [10, 50, 100], 'criterion': ['gini', 'entropy']}, preprocess_default

