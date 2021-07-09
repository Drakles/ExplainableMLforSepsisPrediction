import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sktime.classification.compose import TimeSeriesForestClassifier, \
    ColumnEnsembleClassifier
from sktime.transformers.series_as_features.compose import ColumnConcatenator
from sktime_dl.deeplearning import EncoderClassifier, MCDCNNClassifier
from sktime_dl.utils import check_and_clean_data
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from lstm_autoencoder import one_hot_encode
from prepare_dataset import prepare_dataset
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, \
    recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


def on_epoch_end(self, epoch, logs={}):
    val_predict = (
        np.asarray(self.model.predict(self.model.validation_data[0]))).round()
    val_targ = self.model.validation_data[1]
    _val_f1 = f1_score(val_targ, val_predict)
    _val_recall = recall_score(val_targ, val_predict)
    _val_precision = precision_score(val_targ, val_predict)
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    print(" — val_f1: % f — val_precision: % f — val_recall % f", _val_f1,
          _val_precision, _val_recall)
    return


def get_train_test_dataset(non_sepsis_df, sepsis_df):
    X = non_sepsis_df.append(sepsis_df)
    y = np.array(['non_sepsis' for i in range(len(non_sepsis_df))] +
                 ['sepsis' for i in range(len(sepsis_df))], dtype=object)
    return (train_test_split(X, y, random_state=2137)), X, y


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


def MCDCNN():
    es = EarlyStopping(monitor='val_loss', mode='min',
                       verbose=1, patience=50, restore_best_weights=True)
    filename = '/models/' + MCDCNNClassifier.__name__ + '.h5'
    mc = ModelCheckpoint(filename, monitor='val_loss', save_best_only=True,
                         verbose=True)
    metrics_callback = Metrics()
    model = MCDCNNClassifier(nb_epochs=100, verbose=True,
                             callbacks=[es, mc, metrics_callback])
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
