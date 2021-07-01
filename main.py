import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_dataset(df):
    patient_IDs = set(df['PatientID'])

    patients = np.array()
    for patient_ID in patient_IDs:
        patient_id_df = df[df['PatientID'] == patient_ID]
        patient_id_df.set_index('TimeStampScaled')
        patient_nd_arr = np.array(patient_id_df)
        np.append(patients, patient_nd_arr,axis=0)

    return patients


def top_n_features_with_least_nan(df, n):
    print(df.isna().sum().sort_values()[5:5+n])


if __name__ == '__main__':
    non_sepsis_series_df = pd.read_csv('data/FinalNonSepsisSeries.csv')

    # print(top_n_features_with_least_nan(non_sepsis_series_df,n=5))

    non_sepsis_series_df['TimeStamp'] = non_sepsis_series_df['Day'] \
                                            .astype(int) \
                                        * 24 + \
                                        non_sepsis_series_df['OrdinalHour'] \
                                            .astype(int)


    # non_sepsis_series_df['TimeStampScaled'] = MinMaxScaler().fit_transform(
    #     np.array(non_sepsis_series_df['TimeStamp']).reshape(-1, 1))

    # dataset = prepare_dataset(non_sepsis_series_df)
