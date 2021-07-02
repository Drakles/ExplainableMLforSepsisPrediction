from sktime.datasets import load_basic_motions
import numpy as np
import pandas as pd

from prepare_dataset import prepare_dataset

if __name__ == '__main__':
    # X, y = load_basic_motions(return_X_y=True)
    # df = load_basic_motions()
    non_sepsis_raw_df = pd.read_csv('data/FinalNonSepsisSeries.csv')
    non_sepsis_df = prepare_dataset(non_sepsis_raw_df)

    sepsis_raw_df = pd.read_csv('data/FinalSepsisSeries.csv')
    sepsis_df = prepare_dataset(sepsis_raw_df)

    X = non_sepsis_df.append(sepsis_df)
    y = np.array(['non_sepsis' for i in range(len(non_sepsis_df))] + [
        'sepsis' for i in range(len(sepsis_df))], dtype=object)
