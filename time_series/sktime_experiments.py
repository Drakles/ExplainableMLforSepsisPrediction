import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sktime.classification.compose import TimeSeriesForestClassifier, \
    ColumnEnsembleClassifier
from sktime.transformers.series_as_features.compose import ColumnConcatenator

from time_series.prepare_dataset import prepare_time_series_dataset, \
    summary_missing_val
from utils import get_train_test_time_series_dataset


def read_prepare_series_dataset():
    non_sepsis_raw_df = pd.read_csv('./data/FinalNonSepsisSeries.csv')
    sepsis_raw_df = pd.read_csv('./data/FinalSepsisSeries.csv')

    # print('missing for non sepsis')
    # summary_missing_val(non_sepsis_raw_df)
    #
    # print('missing for sepsis')
    # summary_missing_val(sepsis_raw_df)

    columns_to_drop = ['PatientID', 'Day', 'OrdinalHour', 'Mortality14Days',
                       'comorbidity', 'Admit Ht']

    # shared features
    columns = sorted(list(set(sepsis_raw_df.columns.values)
                          .intersection(non_sepsis_raw_df.columns.values)))
    columns = [col for col in columns if col not in columns_to_drop]
    columns.insert(0, 'PatientID')

    series_non_sepsis_df = prepare_time_series_dataset(non_sepsis_raw_df,
                                                       len(non_sepsis_raw_df.columns),
                                                       columns)

    series_sepsis_df = prepare_time_series_dataset(sepsis_raw_df,
                                                   len(sepsis_raw_df.columns),
                                                   columns)

    # exclude patients from non sepsis if they are in sepsis file
    series_non_sepsis_df = series_non_sepsis_df[
        ~series_non_sepsis_df.index.isin(series_sepsis_df.index.values)]

    return series_non_sepsis_df, series_sepsis_df


def column_concatenate_clf(X_train, y_train):
    steps = [
        ("concatenate", ColumnConcatenator()),
        ("classify", TimeSeriesForestClassifier(class_weight='balanced',
                                                verbose=True,
                                                n_jobs=-1)),
    ]
    clf = Pipeline(steps, verbose=True)
    clf.fit(X_train, y_train)
    return clf


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
    series_non_sepsis_df, series_sepsis_df = read_prepare_series_dataset()

    (X_train, X_test, y_train,
     y_test), X, y = get_train_test_time_series_dataset(
        series_non_sepsis_df,
        series_sepsis_df)

    model = column_ensemble(X_train, y_train, len(X.columns))
    # print(model.score(X_test, y_test))
    print('f1 score: ' + str(f1_score(y_test, model.predict(X_test),
                                      average='weighted')))
    df_pred = pd.DataFrame(data={'TSPred': model.predict_proba(X).T[1],
                                 'PatientID': np.array(X.index, dtype=int)})

    return df_pred, X, y


def fit_predict_time_series_separate_classification():
    series_non_sepsis_df, series_sepsis_df = read_prepare_series_dataset()

    X = series_non_sepsis_df.append(series_sepsis_df)
    y = np.array(['non_sepsis' for _ in range(len(series_non_sepsis_df))] +
                 ['sepsis' for _ in range(len(series_sepsis_df))], dtype=object)

    predictions_per_feature = {}
    predictions_per_feature['PatientID'] = np.array(X.index, dtype=int)

    for f_index in range(len(X.columns)):
        X_one_column = pd.DataFrame(X.iloc[:, f_index])
        X_train, X_test, y_train, y_test = train_test_split(X_one_column, y,
                                                            random_state=2137)
        feature_name = str(X.columns[f_index]) \
            .replace('[', '-') \
            .replace(']', '')
        clf = TimeSeriesForestClassifier(n_estimators=5,
                                         class_weight='balanced')
        clf.fit(X_train, y_train)
        # print('feature: ' + feature_name)
        # print(clf.score(X_test, y_test))
        # print('f1 score: ' + str(f1_score(y_test, clf.predict(X_test),
        #                                   average='weighted')))

        predictions_per_feature[feature_name] = clf.predict_proba(
            X_one_column).T[1]

    return pd.DataFrame(data=predictions_per_feature), X, y


if __name__ == '__main__':
    pred, X, y = fit_predict_time_series_separate_classification()
    # pred, X,y = fit_predict_time_series_column_ensemble()
