from models.model_utils import load_data, preprocess_data
from models.hyperparam_tuning import get_best_model
from models.model_utils import get_knn,get_svm,get_gaussian_nb,get_bernoulli_nb,get_categorical_nb,get_decision_tree,get_random_forest,get_complement_nb,get_multinomial_nb,get_logistic_regression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix,classification_report
import pickle

# Define Model Configurations
param_grid = {
    "Logistic Regression": get_logistic_regression,
    "KNN": get_knn,
    "Gaussian NB": get_gaussian_nb,
    "Multinomial NB": get_multinomial_nb,
    "Bernoulli NB": get_bernoulli_nb,
    "Complement NB": get_complement_nb,
    "SVM": get_svm,
    "Decision Tree": get_decision_tree,
    "Random Forest": get_random_forest
}


# Load & Preprocess Data
dataset = load_data("../data/CKD.csv")
print(dataset.columns)
X_train, X_test, y_train, y_test = preprocess_data(dataset, target_column="classification_yes")

# Training and Comparing Models
best_model_name = None
best_model = None
best_accuracy = 0
best_params = None

for model_name, model_func in param_grid.items():
    print(f"\nTraining {model_name}...")
    model, hyperparams, preprocess_func = model_func()

    # Apply the correct preprocessing function BEFORE hyperparameter tuning
    X_train_preprocessed, X_test_preprocessed = preprocess_func(X_train, X_test)

    # Perform hyperparameter tuning if needed
    best_model_instance, best_params = get_best_model(model, hyperparams, X_train_preprocessed, y_train) if hyperparams else (model, None)

    # Fit the best model
    best_model_instance.fit(X_train_preprocessed, y_train)

    # Predictions
    y_pred = best_model_instance.predict(X_test_preprocessed)
    y_prob = best_model_instance.predict_proba(X_test_preprocessed)[:, 1] if hasattr(best_model_instance, "predict_proba") else None


    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}" if roc_auc else "ROC AUC: Not Available")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test,y_pred)}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = best_model_instance
print(f"\nBest Model: {best_model_name}")
print(f"Best Model Parameters: {best_params} with accuracy: {best_accuracy}")

print("Training Features Shape:", X_train.shape)

# Save Best Model
pickle.dump(best_model, open("../data/best_model_CKD.sav", "wb"))
print(f"\n Best Model Saved: {best_model_name} with Accuracy: {best_accuracy:.4f}")
