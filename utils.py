import numpy as np
from sklearn.model_selection import train_test_split


def get_train_test_dataset(non_sepsis_df, sepsis_df):
    X = non_sepsis_df.append(sepsis_df)
    y = np.array(['non_sepsis' for i in range(len(non_sepsis_df))] +
                 ['sepsis' for i in range(len(sepsis_df))], dtype=object)
    return (train_test_split(X, y, random_state=2137)), X, y
