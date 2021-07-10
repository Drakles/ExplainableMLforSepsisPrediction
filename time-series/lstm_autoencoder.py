import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sktime_dl.utils import check_and_clean_data

from prepare_dataset import prepare_dataset


def one_hot_encode(y):
    label_encoder = LabelEncoder()
    # categories='auto' to get rid of FutureWarning
    onehot_encoder = OneHotEncoder(sparse=False, categories="auto")

    y = label_encoder.fit_transform(y)
    classes_ = label_encoder.classes_
    nb_classes = len(classes_)

    y = y.reshape(len(y), 1)
    y = onehot_encoder.fit_transform(y)
    return y


def get_model(input_shape):
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=input_shape,
                   return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=False))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[1])))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss="mse")

    return model


if __name__ == '__main__':
    non_sepsis_raw_df = pd.read_csv('../data/FinalNonSepsisSeries.csv')
    non_sepsis_df = prepare_dataset(non_sepsis_raw_df)

    # non_sepsis_df = prepare_dataset_lstm_autoencoder(non_sepsis_raw_df)
    # sepsis_raw_df = pd.read_csv('data/FinalSepsisSeries.csv')
    # sepsis_df = prepare_dataset_lstm_autoencoder(sepsis_raw_df)

    # X = non_sepsis_df.append(sepsis_df)
    # y = np.array(['non_sepsis' for i in range(len(non_sepsis_df))] +
    #              ['sepsis' for i in range(len(sepsis_df))], dtype=object)

    X = non_sepsis_df
    y = np.array(['non_sepsis' for i in range(len(non_sepsis_df))])

    X_sktime = check_and_clean_data(X, input_checks=True)
    y_one_hot = one_hot_encode(y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2137)

    input_shape = X_sktime.shape[1:]
    model = get_model(input_shape)
    model.fit(X_sktime, X_sktime, epochs=10, validation_split=0.1)
