import pandas as pd
from sklearn.impute import KNNImputer

pd.options.mode.chained_assignment = None


def summary_missing_val(dataframe):
    # summarize the number of rows with missing values for each column
    percent_missing = []
    print('Column name, Number of values, Number of missing, % of missing')
    for col in dataframe:
        # count number of rows with missing values
        n_miss = dataframe[col].isnull().sum()
        total_val = len(dataframe[col]) - n_miss
        perc = n_miss / dataframe.shape[0] * 100
        print('%s ,%d, %d, %.1f%%' % (col, total_val, n_miss, perc))
        percent_missing.append(perc)
    print('% of missing:' + str(sum(percent_missing) / len(percent_missing)))


def prepare_time_series_dataset(df, selected_features):
    # filter for selected columns
    df = df[selected_features]

    patientIDs = set(df['PatientID'])

    for patientID in patientIDs:
        df.loc[df['PatientID'] == patientID, selected_features[1:]] = \
            df[df['PatientID'] == patientID][selected_features[1:]] \
                .interpolate(method='pad', limit_direction='forward') \
                .interpolate(method='bfill')

    df = knn_imputing(df)

    patient_id_series = {}

    for id, patient_id_df in df.groupby('PatientID'):
        patient_id_series[id] = [id] + [patient_id_df[col_series]
                                            .tail(24)
                                            .reset_index(drop=True)
                                        for col_series in selected_features[1:]]

    resultd_df = pd.DataFrame. \
        from_dict(patient_id_series, "index", columns=df.columns) \
        .set_index('PatientID')

    return resultd_df


def knn_imputing(df):
    imputer = KNNImputer(n_neighbors=5, weights='distance',
                         metric='nan_euclidean')
    Xtrans = imputer.fit_transform(df)
    df = pd.DataFrame(data=Xtrans, columns=df.columns)
    return df


if __name__ == '__main__':
    df_series_non_sepsis = pd.read_csv('../data/FinalNonSepsisSeries.csv')
    df_series_sepsis = pd.read_csv('../data/FinalSepsisSeries.csv')

    columns_to_drop = ['PatientID', 'Day', 'OrdinalHour', 'Mortality14Days',
                       'comorbidity', 'Admit Ht']

    # shared features
    columns = sorted(list(set(df_series_sepsis.columns.values)
        .intersection(
        set(df_series_non_sepsis.columns.values))))
    columns = [col for col in columns if col not in columns_to_drop]
    columns.insert(0, 'PatientID')

    series_non_sepsis_df = prepare_time_series_dataset(df_series_non_sepsis,
                                                       columns)

    series_sepsis_df = prepare_time_series_dataset(df_series_sepsis,
                                                   columns)

    # exclude patients from non sepsis if they are in sepsis file
    series_non_sepsis_df = series_non_sepsis_df[
        ~series_non_sepsis_df.index.isin(series_sepsis_df.index.values)]

    series_non_sepsis_df.to_pickle(
        '../data/preprocessed_data/series_non_sepsis.pkl')
    series_sepsis_df.to_pickle('../data/preprocessed_data/series_sepsis.pkl')

