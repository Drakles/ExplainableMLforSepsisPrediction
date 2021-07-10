import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sktime.classification.compose import TimeSeriesForestClassifier, \
    ColumnEnsembleClassifier
from sktime.transformers.series_as_features.compose import ColumnConcatenator
from sktime_dl.deeplearning import MCDCNNClassifier
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

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


def column_ensemble(X_train, y_train):
    steps = [('classify', ColumnEnsembleClassifier(
        estimators=get_estimators(nb_features=5)),)]
    clf = Pipeline(steps)
    clf.fit(X_train, y_train)

    return clf


def MCDCNN():
    es = EarlyStopping(monitor='val_loss', mode='min',
                       verbose=1, patience=50, restore_best_weights=True)
    filename = '/models/' + MCDCNNClassifier.__name__ + '.h5'
    mc = ModelCheckpoint(filename, monitor='val_loss', save_best_only=True,
                         verbose=True)
    model = MCDCNNClassifier(nb_epochs=100, verbose=True,
                             callbacks=[es, mc])
    model.fit(X_train, y_train)

    return model


def fit_predict_time_series():
    series_non_sepsis_df, series_sepsis_df = read_prepare_series_dataset()

    (X_train, X_test, y_train,
     y_test), X, y = get_train_test_time_series_dataset(
        series_non_sepsis_df,
        series_sepsis_df)

    model = column_ensemble(X_train, y_train)
    # print(model.score(X_test, y_test))
    df_pred = pd.DataFrame(data={'TSPred': model.predict(X),
                                 'PatientID': np.array(X.index, dtype=int)})

    return df_pred, X, y


if __name__ == '__main__':
    non_sepsis_df, sepsis_df = read_prepare_series_dataset()

    (X_train, X_test, y_train,
     y_test), X, y = get_train_test_time_series_dataset(
        non_sepsis_df,
        sepsis_df)

    model = column_ensemble(X_train, y_train)
    print(model.score(X_test, y_test))
