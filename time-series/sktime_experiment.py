import pandas as pd
from sklearn.pipeline import Pipeline
from sktime.classification.compose import TimeSeriesForestClassifier, \
    ColumnEnsembleClassifier
from sktime.transformers.series_as_features.compose import ColumnConcatenator
from sktime_dl.deeplearning import MCDCNNClassifier
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from prepare_dataset import prepare_dataset
from utils import get_train_test_dataset


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
                    n_estimators=50,
                    class_weight='balanced',
                    verbose=True), [i])
        )
    return estimators


def column_ensemble(X_train, y_train):
    steps = [('classify', ColumnEnsembleClassifier(
                 estimators=get_estimators(nb_features=5)),
              )]
    clf = Pipeline(steps, verbose=True)
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


if __name__ == '__main__':
    non_sepsis_raw_df = pd.read_csv('data/FinalNonSepsisSeries.csv')
    non_sepsis_df = prepare_dataset(non_sepsis_raw_df)
    sepsis_raw_df = pd.read_csv('data/FinalSepsisSeries.csv')
    sepsis_df = prepare_dataset(sepsis_raw_df)

    (X_train, X_test, y_train, y_test), X, y = get_train_test_dataset(
        non_sepsis_df,
        sepsis_df)

    # model = column_concatenate_clf(X_train, y_train)
    model = column_ensemble(X_train, y_train)
    print(model.score(X_test, y_test))
