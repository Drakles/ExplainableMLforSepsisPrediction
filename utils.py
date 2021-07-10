import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_time_series_dataset(non_sepsis_df, sepsis_df):
    X = non_sepsis_df.append(sepsis_df)
    # if df_ts_pred is not None:
    #     indices_from_static = np.unique(np.append(np.array(non_sepsis_df.index),
    #                                               np.array(sepsis_df.index)))
    #     # df_ts_pred = df_ts_pred.drop(indices_from_static)
    #     df_ts_pred = df_ts_pred[df_ts_pred['PatientID']
    #         .isin(indices_from_static)]
    #     df_ts_pred = df_ts_pred.set_index('PatientID')
    #
    #     X = pd.merge(X, df_ts_pred, left_index=True, right_index=False)

    y = np.array(['non_sepsis' for _ in range(len(non_sepsis_df))] +
                 ['sepsis' for _ in range(len(sepsis_df))], dtype=object)
    return (train_test_split(X, y, random_state=2137)), X, y
