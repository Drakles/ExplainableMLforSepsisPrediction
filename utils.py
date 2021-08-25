import numpy as np
from sklearn.model_selection import train_test_split


def get_train_test_time_series_dataset(non_sepsis_df, sepsis_df):
    X = non_sepsis_df.append(sepsis_df)
    y = np.array(['non_sepsis' for _ in range(len(non_sepsis_df))] +
                 ['sepsis' for _ in range(len(sepsis_df))], dtype=object)
    return (train_test_split(X, y, random_state=2137, stratify=y)), X, y


def merge_static_series_pred(df_static_non_sepsis,
                             df_static_sepsis,
                             df_ts_pred):
    X_static = df_static_non_sepsis.append(df_static_sepsis)

    # with feature removing
    df_ts_pred = df_ts_pred.drop(['LDH-TSP', 'Daily Weight-TSP', 'AST-TSP'],
                                 axis=1)
    X_static = X_static.drop(
        ['los', 'respiration', 'sofa'], axis=1)

    X = X_static.merge(df_ts_pred, how='left')

    X = X.set_index('PatientID')

    y = X['Label']
    X = X.drop('Label', axis=1)

    return X, y
