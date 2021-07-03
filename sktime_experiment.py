from sklearn.model_selection import train_test_split
from sktime.datasets import load_basic_motions
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.datasets import load_basic_motions
from sktime.transformations.panel.compose import ColumnConcatenator

from prepare_dataset import prepare_dataset

if __name__ == '__main__':
    # X, y = load_basic_motions(return_X_y=True)
    # df = load_basic_motions()

    non_sepsis_raw_df = pd.read_csv('data/FinalNonSepsisSeries.csv')
    non_sepsis_df = prepare_dataset(non_sepsis_raw_df)

    sepsis_raw_df = pd.read_csv('data/FinalSepsisSeries.csv')
    sepsis_df = prepare_dataset(sepsis_raw_df)

    X = non_sepsis_df.append(sepsis_df)
    y = np.array(['non_sepsis' for i in range(len(non_sepsis_df))] +
                 ['sepsis' for i in range(len(sepsis_df))], dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2137)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    #first approach
    steps = [
        ("concatenate", ColumnConcatenator()),
        ("classify", TimeSeriesForestClassifier(n_estimators=100)),
    ]
    clf = Pipeline(steps)
    clf.fit(X_train, y_train)

    #second
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
    clf.score(X_test, y_test)

    #third
    clf = MrSEQLClassifier()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
