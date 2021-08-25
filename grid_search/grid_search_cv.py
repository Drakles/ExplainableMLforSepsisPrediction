from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
from sklearn.utils import class_weight


def custom_score(model, X, y):
    return f1_score(y, model.predict(X), average='weighted') + \
           roc_auc_score(y, model.predict_proba(X)[:, 1])


def search_params(X, y, param_grid):
    CV = GridSearchCV(xgb.XGBClassifier(random_state=2137), param_grid,
                      n_jobs=-1,
                      cv=StratifiedKFold(), scoring=custom_score, verbose=1)
    CV.fit(X, y, sample_weight=class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y))

    print(CV.best_params_)
    print(CV.best_score_)

    return CV


if __name__ == '__main__':
    param_grid = {
        'n_estimators': range(180, 300),
        'max_depth': range(4, 15)
    }
    CV = search_params(X, y, param_grid)
