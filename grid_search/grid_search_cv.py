from sklearn.model_selection import GridSearchCV
import xgboost as xgb


def search_params(X, classes_weights, X_train, y_train):
    param_grid = {
        'n_estimators': [200, 250, 300, 350],
        'max_depth': [X.columns.shape[0] // 4,
                      X.columns.shape[0] // 3, X.columns.shape[0] // 2],
        'objective': ['binary:logistic'],
        'booster': ['gbtree', 'gblinear', 'dart'],
        'n_jobs': [4]}

    CV = GridSearchCV(xgb.XGBClassifier(), param_grid, n_jobs=-1)
    CV.fit(X_train, y_train, sample_weight=classes_weights)

    print(CV.best_params_)

    return CV.best_estimator_
