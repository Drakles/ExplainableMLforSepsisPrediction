import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sktime.classification.compose import TimeSeriesForestClassifier, \
    ColumnEnsembleClassifier
from sktime.transformers.series_as_features.compose import ColumnConcatenator

from time_series.prepare_dataset import prepare_time_series_dataset
from utils import get_train_test_time_series_dataset


def read_prepare_series_dataset():
    non_sepsis_raw_df = pd.read_csv('data/FinalNonSepsisSeries.csv')
    series_non_sepsis_df = prepare_time_series_dataset(non_sepsis_raw_df)
    sepsis_raw_df = pd.read_csv('data/FinalSepsisSeries.csv')
    series_sepsis_df = prepare_time_series_dataset(sepsis_raw_df)

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
                    n_estimators=1,
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

    model = column_ensemble(X_train, y_train, 5)
    # print(model.score(X_test, y_test))
    df_pred = pd.DataFrame(data={'TSPred': model.predict(X),
                                 'PatientID': np.array(X.index, dtype=int)})

    return df_pred, X


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
        # print('feature: ' + feature_name)
        clf = TimeSeriesForestClassifier(n_estimators=1,
                                         class_weight='balanced')
        clf.fit(X_train, y_train)
        # print(clf.score(X_test, y_test))

        predictions_per_feature[feature_name] = clf.predict(X_one_column)

    return pd.DataFrame(data=predictions_per_feature), X


if __name__ == '__main__':
    pred, X = fit_predict_time_series_separate_classification()
