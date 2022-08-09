import pandas as pd
from sklearn.impute import KNNImputer

pd.options.mode.chained_assignment = None


def summary_missing_val(dataframe):
    # summarize the number of rows with missing values for each column
    percent_missing = []
    print('Column name, Number of values, Number of missing, % of missing')
    for col in dataframe:
        n_miss = dataframe[col].isnull().sum()
        total_val = len(dataframe[col]) - n_miss
        perc = n_miss / dataframe.shape[0] * 100
        print('%s ,%d, %d, %.1f%%' % (col, total_val, n_miss, perc))
        percent_missing.append(perc)
    print('% of missing:' + str(sum(percent_missing) / len(percent_missing)))


def process_time_series_dataset(df, selected_features):
    # filter by selected columns
    df = df[['PatientID'] + selected_features + ['Day', 'OrdinalHour']]

    df['TimeStamp'] = df.Day.astype(str).str.cat(df.OrdinalHour.astype(str),
                                                 sep='-')
    df = df.drop(['Day', 'OrdinalHour'], axis=1)

    patientIDs = set(df['PatientID'])

    for patientID in patientIDs:
        if df[df['PatientID'] == patientID]['TimeStamp'].is_unique:
            # interpolate
            df.loc[df['PatientID'] == patientID, selected_features] = \
                df[df['PatientID'] == patientID][selected_features] \
                    .interpolate(method='pad', limit_direction='forward') \
                    .interpolate(method='bfill')
        else:
            # drop repeated ids
            df = df.drop(df[df['PatientID'] == patientID].index)

    df = df.drop('TimeStamp', axis=1)

    # impute values with KNNImputer
    df = knn_imputing(df)

    patient_id_series = {}

    # change data format to 3D dataframe and limit time series to last 24 hours
    for id, patient_id_df in df.groupby('PatientID'):
        patient_id_series[id] = [id] + [patient_id_df[sel_col]
                                            .tail(24)
                                            .reset_index(drop=True)
                                        for sel_col in selected_features]

    result_df = pd.DataFrame. \
        from_dict(patient_id_series, "index", columns=df.columns) \
        .set_index('PatientID')

    return result_df


def knn_imputing(df):
    imputer = KNNImputer(n_neighbors=5, weights='distance',
                         metric='nan_euclidean')
    Xtrans = imputer.fit_transform(df)
    df = pd.DataFrame(data=Xtrans, columns=df.columns)
    return df


def save_df_intersection_features(df_series_sepsis, df_series_non_sepsis,
                                  columns_to_drop):
    features_intersection = sorted(list(set(df_series_sepsis.columns.values)
        .intersection(
        set(df_series_non_sepsis.columns.values))))
    selected_columns = [feature for feature in features_intersection
                        if feature not in columns_to_drop]
    series_non_sepsis_df = process_time_series_dataset(df_series_non_sepsis,
                                                       selected_columns)
    series_sepsis_df = process_time_series_dataset(df_series_sepsis,
                                                   selected_columns)

    series_non_sepsis_df.to_pickle(
        '../data/preprocessed_data/union_features/series_non_sepsis.pkl')
    series_sepsis_df.to_pickle(
        '../data/preprocessed_data/union_features/series_sepsis.pkl')


if __name__ == '__main__':
    df_series_non_sepsis = pd.read_csv('../data/FinalNonSepsisSeries.csv')
    df_series_sepsis = pd.read_csv('../data/FinalSepsisSeries.csv')

    columns_to_drop = ['PatientID', 'Mortality14Days', 'Day', 'OrdinalHour',
                       'comorbidity', 'Admit Ht', 'Admission Weight (Kg)',
                       'Admission Weight (lbs.)', 'Height (cm)', 'Height',
                       'Hour', 'weight', 'Age']

    # use only shared features
    save_df_intersection_features(df_series_sepsis, df_series_non_sepsis,
                                  columns_to_drop)
