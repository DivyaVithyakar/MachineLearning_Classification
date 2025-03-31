from sklearn.model_selection import GridSearchCV

# GridSearch Hyperparameter Tuning
def get_best_model(model, param_grid, X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV and return the best model & parameters.
    """
    if not param_grid:  # If no hyperparameters, just return the model as is.
        return model, {}

    grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", refit=True, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_