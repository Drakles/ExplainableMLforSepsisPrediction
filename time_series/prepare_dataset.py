import pandas as pd
from sklearn.impute import KNNImputer


def summary_missing_val(dataframe):
    # summarize the number of rows with missing values for each column
    for col in dataframe:
        # count number of rows with missing values
        n_miss = dataframe[col].isnull().sum()
        perc = n_miss / dataframe.shape[0] * 100
        print('> column %s, Missing: %d (%.1f%%)' % (col, n_miss, perc))


def prepare_time_series_dataset(df, nb_features, selected_features):
    # df with top n features with the least number of nan values
    if len(df.columns) != nb_features:
        df = df[top_n_features_with_least_nan(df, nb_features)]

    # filter for selected columns
    df = df[selected_features]

    # drop all columns with no features
    df = df.dropna(axis=1, how='all')

    for column in selected_features[1:]:
        df[column] = df[column] \
            .interpolate(method='pad', limit_direction='forward') \
            .interpolate(method='bfill')

    df = knn_imputing(df)

    patient_id_series = {}

    for id, patient_id_df in df.groupby('PatientID'):
        patient_id_series[id] = [id] + \
                                [patient_id_df[col_series]
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


def top_n_features_with_least_nan(df, n):
    return df.isna().sum().sort_values()[0:n].index.values


if __name__ == '__main__':
    df = pd.read_csv('../data/FinalNonSepsisSeries.csv')

    dataset = prepare_time_series_dataset(df, df.columns.values)
    print(dataset.describe())
