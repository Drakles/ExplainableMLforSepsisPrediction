import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_dataset(df):
    patient_IDs = np.unique(df['PatientID'])
    feature_columns = df.columns[1:].values

    patients = pd.DataFrame(data=[], columns=df.columns)
    for patient_ID in patient_IDs:
        patient_id_df = df[df['PatientID'] == patient_ID]
        columns_series = [patient_id_df[col_series] for col_series
                          in feature_columns]
        patients = patients.append(pd.DataFrame(np.array([patient_ID] +
                                                       columns_series,
                                                         dtype=object)
                                     .reshape(1, 5),
                                     columns=df.columns))

    return patients.set_index('PatientID')


def top_n_features_with_least_nan(df, n):
    return df.isna().sum().sort_values()[5:5 + n].index.values


if __name__ == '__main__':
    df = pd.read_csv('data/FinalNonSepsisSeries.csv')

    # print(top_n_features_with_least_nan(non_sepsis_series_df,n=5))

    df['TimeStamp'] = df['Day'].astype(int) * 24 + df['OrdinalHour'].astype(int)

    # df with top n features with the least number of nan values
    df = df[df.isna().sum().sort_values()[0:10].index.values]
    # drop irrelevant features
    df = df.drop(
        columns=['comorbidity', 'Mortality14Days', 'Day', 'OrdinalHour',
                 'TimeStamp'])

    # non_sepsis_series_df['TimeStampScaled'] = MinMaxScaler().fit_transform(
    #     np.array(non_sepsis_series_df['TimeStamp']).reshape(-1, 1))

    dataset = prepare_dataset(df)
    print(dataset.describe())
