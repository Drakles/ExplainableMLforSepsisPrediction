import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.random import normal
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_validate, \
    StratifiedKFold, cross_val_predict
from sklearn.utils import class_weight

from time_series.sktime_hybrid import \
    fit_predict_time_series_hybrid_classification


def sample_old_age_with_distribution(df):
    np.random.seed(2137)
    data = normal(loc=65.79, scale=16.59, size=len(df['Age']))
    data = [age for age in data if age >= 89]
    limit = len(df['Age'].loc[df['Age'] > 89])
    data = sorted(data)[:limit]

    df.loc[df['Age'] > 89, ['Age']] = data
    df['Age'] = df['Age'].astype(int)

    return df


def age_histogram(df, bins):
    _, _, _ = plt.hist(df['Age'], bins=bins)

    plt.xlabel('Age of the patient')
    plt.ylabel('Count')
    plt.legend(loc="lower right")

    plt.show()


def read_prepare_static_data():
    df_static_sepsis = pd.read_csv('data/FinalSepsisCohort.csv')
    df_static_non_sepsis = pd.read_csv('data/FinalNonSepsisCohort.csv')

    df_static_non_sepsis['Label'] = np.array(['non_sepsis' for _ in range(
        len(df_static_non_sepsis))])

    df_static_sepsis['Label'] = np.array(['sepsis' for _ in range(
        len(df_static_sepsis))])

    df = df_static_non_sepsis.append(df_static_sepsis)

    # histogram of the age
    # age_histogram(df, 100)

    df = df.drop('Deathperiod', axis='columns')

    # reconstruct the age distribution
    df = sample_old_age_with_distribution(df)

    # remove duplicate IDs
    df = df.drop_duplicates(subset='PatientID', keep=False)

    # histogram of the age
    # age_histogram(df, 79)

    df_static_sepsis = df.loc[df['Label'] == 'sepsis']
    df_static_non_sepsis = df.loc[df['Label'] == 'non_sepsis']

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


def merge_static_series_pred(df_static_non_sepsis,
                             df_static_sepsis,
                             df_ts_pred):
    X_static = df_static_non_sepsis.append(df_static_sepsis)

    # with feature removing
    df_ts_pred = df_ts_pred.drop(['LDH-TSP', 'Daily Weight-TSP', 'AST-TSP'],
                                 axis=1)
    X_static = X_static.drop(
        ['LOS', 'Respiration', 'Coagulation'], axis=1)

    X = X_static.merge(df_ts_pred, how='left')

    X = X.set_index('PatientID')

    y = X['Label']
    X = X.drop('Label', axis=1)

    return X, y


def get_xgboost_X_enhanced():
    df_static_non_sepsis, df_static_sepsis = read_prepare_static_data()
    df_ts_pred = fit_predict_time_series_hybrid_classification(
        './data/preprocessed_data/union_features/series_sepsis.pkl',
        './data/preprocessed_data/union_features/series_non_sepsis.pkl')
    X, y = merge_static_series_pred(df_static_non_sepsis,
                                    df_static_sepsis,
                                    df_ts_pred)

    fit_param = {'sample_weight': class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y
    )}
    model_param = {'max_depth': 8,
                   'objective': 'binary:logistic',
                   'n_estimators': 182, 'booster': 'gbtree',
                   'random_state': 2137}

    model = xgb.XGBClassifier(**model_param)

    scores = cross_validate(model, X, y,
                            scoring=['f1_weighted', 'roc_auc'], verbose=1,
                            cv=StratifiedKFold(), fit_params=fit_param)

    print('avg f1 score:' + str(np.mean(scores['test_f1_weighted'])))
    print('avg roc auc score: ' + str(np.mean(scores['test_roc_auc'])))

    # plot_roc_auc(y, cross_val_predict(model, X, y, cv=StratifiedKFold(),
    #                                   method='predict_proba'))

    model.fit(X, y, sample_weight=fit_param['sample_weight'])

    return model, X, y


def plot_tree(model):
    xgb.plot_tree(model)
    plt.show()


if __name__ == '__main__':
    model, X, y = get_xgboost_X_enhanced()
