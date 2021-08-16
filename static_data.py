import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from time_series.sktime_column_ensemble import \
    fit_predict_time_series_separate_classification
from utils import merge_static_series_pred


def read_prepare_static_data():
    df_static_sepsis = pd.read_csv('data/FinalSepsisCohort.csv')
    df_static_non_sepsis = pd.read_csv('data/FinalNonSepsisCohort.csv')

    df_static_non_sepsis = df_static_non_sepsis.drop('deathperiod',
                                                     axis='columns')
    # remove duplicate IDs
    df_static_non_sepsis = df_static_non_sepsis. \
        drop_duplicates(subset='PatientID', keep=False)

    df_static_sepsis = df_static_sepsis.drop('deathperiod',
                                             axis='columns')
    # remove duplicate IDs
    df_static_sepsis = df_static_sepsis. \
        drop_duplicates(subset='PatientID', keep=False)

    if not set(df_static_sepsis['PatientID']).isdisjoint(set(
            df_static_non_sepsis['PatientID'])):
        raise ValueError('overlapping patients id in short and series')

    return df_static_non_sepsis, df_static_sepsis


def plot_roc_auc(y_test, predictions):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test, predictions[:, i],
                                      pos_label='sepsis')
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(),
                                              predictions.T[-1],
                                              pos_label='sepsis')
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for XGBoost')
    plt.legend(loc="lower right")
    plt.savefig('output/roc_auc.png')
    plt.show()


def get_xgboost_X_enhanced():
    df_static_non_sepsis, df_static_sepsis = read_prepare_static_data()
    df_ts_pred = \
        fit_predict_time_series_separate_classification(
            './data/preprocessed_data/union_features/series_non_sepsis.pkl',
            './data/preprocessed_data/union_features/series_sepsis.pkl')
    X, y = merge_static_series_pred(df_static_non_sepsis,
                                    df_static_sepsis,
                                    df_ts_pred)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=2137,
                                                        stratify=y)

    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    param = {'max_depth': X.columns.shape[0], 'objective': 'binary:logistic',
             'nthread': 4, 'n_estimators': 300, 'booster': 'gbtree'}

    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train, sample_weight=classes_weights)

    print('f1 score: ' + str(f1_score(y_test, model.predict(X_test),
                                      average='weighted')))
    predictions = model.predict_proba(X_test)
    print('roc auc: ' + str(roc_auc_score(y_test, predictions[:, 1])))
    # plot_roc_auc(y_test, predictions)

    return model, X, y


def plot_tree(model):
    xgb.plot_tree(model)
    plt.show()


if __name__ == '__main__':
    model, X, y = get_xgboost_X_enhanced()
