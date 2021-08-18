import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sktime.classification.compose import TimeSeriesForestClassifier, \
    ColumnEnsembleClassifier

from utils import get_train_test_time_series_dataset


def get_estimators(nb_features):
    estimators = []
    for i in range(nb_features):
        estimators.append(
            (
                "TSF" + str(i), TimeSeriesForestClassifier(
                    n_estimators=5,
                    class_weight='balanced'), [i])
        )
    return estimators


def column_ensemble(X_train, y_train, nb_features):
    steps = [('classify', ColumnEnsembleClassifier(
        estimators=get_estimators(nb_features=nb_features)),)]
    clf = Pipeline(steps)
    clf.fit(X_train, y_train)

    return clf


def fit_predict_time_series_column_ensemble():
    series_non_sepsis_df = pd.read_pickle(
        '../data/preprocessed_data/series_non_sepsis.pkl')
    series_sepsis_df = pd.read_pickle(
        '../data/preprocessed_data/series_sepsis.pkl')

    (X_train, X_test, y_train,
     y_test), X, y = get_train_test_time_series_dataset(
        series_non_sepsis_df,
        series_sepsis_df)

    model = column_ensemble(X_train, y_train, len(X.columns))
    print('f1 score: ' + str(f1_score(y_test, model.predict(X_test),
                                      average='weighted')))
    predictions = model.predict_proba(X_test)
    print('roc auc: ' + str(roc_auc_score(y_test, predictions[:, 1])))

    # f1 score: 0.979812122398813
    # roc auc: 0.9979838709677419

    df_pred = pd.DataFrame(data={'TSPred': model.predict_proba(X).T[1],
                                 'PatientID': np.array(X.index, dtype=int)})

    return df_pred, X, y


def fit_predict_time_series_separate_classification(sepsis_df_path,
                                                    non_sepsis_df_path):
    series_non_sepsis_df = pd.read_pickle(non_sepsis_df_path)
    series_non_sepsis_df['Label'] = np.array(['non_sepsis' for _ in range(
        len(series_non_sepsis_df))])
    series_sepsis_df = pd.read_pickle(sepsis_df_path)
    series_sepsis_df['Label'] = np.array(['sepsis' for _ in range(
        len(series_sepsis_df))])

    X = series_non_sepsis_df.append(series_sepsis_df)
    y = X['Label']
    X = X.drop('Label', axis=1)

    predictions_per_feature = {}
    predictions_per_feature['PatientID'] = np.array(X.index, dtype=int)

    f1_scores = []
    roc_auc_scores = []

    for f_index in range(len(X.columns)):
        X_one_column = pd.DataFrame(X.iloc[:, f_index])
        X_train, X_test, y_train, y_test = train_test_split(X_one_column, y,
                                                            random_state=2137,
                                                            stratify=y)
        feature_name = str(X.columns[f_index]) \
                           .replace('[', '-') \
                           .replace(']', '') \
                       + '-TSP'
        model = TimeSeriesForestClassifier(n_estimators=5,
                                           class_weight='balanced')
        model.fit(X_train, y_train)

        # print('feature: ' + feature_name)
        f1_score_val = f1_score(y_test, model.predict(X_test),
                                average='weighted')
        roc_auc_score_val = roc_auc_score(y_test, model.
                                          predict_proba(X_test)[:, 1])
        f1_scores.append(f1_score_val)
        roc_auc_scores.append(roc_auc_score_val)

        # print('f1 score: ' + str(f1_score_val))
        # print('roc auc: ' + str(roc_auc_score_val))

        predictions_per_feature[feature_name] = model.predict_proba(
            X_one_column).T[1]

    # print('f1 average: ' + str(sum(f1_scores) / len(f1_scores)))
    # print('roc auc average: ' + str(sum(roc_auc_scores) / len(roc_auc_scores)))

    return pd.DataFrame(data=predictions_per_feature)


if __name__ == '__main__':
    fit_predict_time_series_separate_classification(
        '../data/preprocessed_data/union_features/series_sepsis.pkl',
        '../data/preprocessed_data/union_features/series_non_sepsis.pkl'
    )
