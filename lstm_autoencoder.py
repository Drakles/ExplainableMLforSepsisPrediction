import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
import tensorflow as tf

from prepare_dataset import top_n_features_with_least_nan, transform_series


def get_model(timesteps, n_features):
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features),
                   return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')

    return model


def prepare_dataset_lstm_autoencoder(df):
    # df with top n features with the least number of nan values
    df = df[top_n_features_with_least_nan(df, 5)]

    # drop irrelevant features
    df = df.drop(
        columns=['comorbidity', 'Mortality14Days', 'Day', 'OrdinalHour'])

    feature_columns = df.columns[1:].values

    patient_id_series = {}

    for id, patient_id_df in df.groupby('PatientID'):
        patient_id_series[id] = \
            [id] + [np.asarray(transform_series(patient_id_df[
                                                    col_series]).to_list(),
                               dtype=object) for col_series in feature_columns]

    return pd.DataFrame. \
        from_dict(patient_id_series, "index", columns=df.columns) \
        .set_index('PatientID')


if __name__ == '__main__':
    non_sepsis_raw_df = pd.read_csv('data/FinalNonSepsisSeries.csv')
    non_sepsis_df = prepare_dataset_lstm_autoencoder(non_sepsis_raw_df)
    sepsis_raw_df = pd.read_csv('data/FinalSepsisSeries.csv')
    sepsis_df = prepare_dataset_lstm_autoencoder(sepsis_raw_df)

    # X = non_sepsis_df.append(sepsis_df)
    # y = np.array(['non_sepsis' for i in range(len(non_sepsis_df))] +
    #              ['sepsis' for i in range(len(sepsis_df))], dtype=object)

    X = non_sepsis_df
    y = np.array(['non_sepsis' for i in range(len(non_sepsis_df))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2137)

    timesteps = len(X_train.iloc[0][0])
    n_features = len(X_train.columns)
    model = get_model(timesteps, n_features)
    # model.fit(X_train, y_train)
