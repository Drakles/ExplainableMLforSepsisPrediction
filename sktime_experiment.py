import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sktime.classification.compose import TimeSeriesForestClassifier, \
    ColumnEnsembleClassifier
from sktime.transformers.series_as_features.compose import ColumnConcatenator
from sktime_dl.deeplearning import EncoderClassifier

from prepare_dataset import prepare_dataset


def get_train_test_dataset(non_sepsis_df, sepsis_df):
    X = non_sepsis_df.append(sepsis_df)
    y = np.array(['non_sepsis' for i in range(len(non_sepsis_df))] +
                 ['sepsis' for i in range(len(sepsis_df))], dtype=object)
    return train_test_split(X, y, random_state=2137)


def column_concatenate_clf(X_train, y_train):
    steps = [
        ("concatenate", ColumnConcatenator()),
        ("classify", TimeSeriesForestClassifier(n_estimators=100)),
    ]
    clf = Pipeline(steps)
    clf.fit(X_train, y_train)
    return clf


def column_ensemble(X_train, y_train):
    clf = ColumnEnsembleClassifier(
        estimators=[
            ("TSF0", TimeSeriesForestClassifier(n_estimators=100), [0]),
            ("TSF01", TimeSeriesForestClassifier(n_estimators=100), [1]),
            ("TSF02", TimeSeriesForestClassifier(n_estimators=100), [2]),
            ("TSF03", TimeSeriesForestClassifier(n_estimators=100), [3]),
            ("TSF04", TimeSeriesForestClassifier(n_estimators=100), [4]),
        ]
    )
    clf.fit(X_train, y_train)

    return clf


if __name__ == '__main__':
    non_sepsis_raw_df = pd.read_csv('data/FinalNonSepsisSeries.csv')
    non_sepsis_df = prepare_dataset(non_sepsis_raw_df)
    sepsis_raw_df = pd.read_csv('data/FinalSepsisSeries.csv')
    sepsis_df = prepare_dataset(sepsis_raw_df)

    X_train, X_test, y_train, y_test = get_train_test_dataset(non_sepsis_df,
                                                              sepsis_df)

    network = EncoderClassifier(nb_epochs=5, verbose=True)
    network.fit(X_train, y_train)
