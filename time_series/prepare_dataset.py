import pandas as pd


def transform_series(series):
    return series \
        .interpolate(method='pad',limit_direction='forward') \
        .interpolate(method='bfill')


def prepare_time_series_dataset(df):
    # df with top n features with the least number of nan values
    df = df[top_n_features_with_least_nan(df, 5)]

    # drop irrelevant features
    df = df.drop(
        columns=['comorbidity', 'Mortality14Days', 'Day', 'OrdinalHour'])

    feature_columns = df.columns[1:].values

    patient_id_series = {}

    for id, patient_id_df in df.groupby('PatientID'):
        patient_id_series[id] = [id] + \
                                [transform_series(patient_id_df[col_series])
                                     .reset_index(drop=True)
                                 for col_series in feature_columns]

    return pd.DataFrame. \
        from_dict(patient_id_series, "index", columns=df.columns) \
        .set_index('PatientID')


def top_n_features_with_least_nan(df, n):
    return df.isna().sum().sort_values()[0:5 + n].index.values


if __name__ == '__main__':
    df = pd.read_csv('../data/FinalNonSepsisSeries.csv')

    dataset = prepare_time_series_dataset(df)
    print(dataset.describe())
