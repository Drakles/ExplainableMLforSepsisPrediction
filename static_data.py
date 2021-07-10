import pandas as pd
import xgboost as xgb

from time_series.sktime_experiment import fit_predict_time_series


def read_prepare_static_data():
    df_sepsis = pd.read_csv('data/FinalSepsisCohort.csv')
    df_non_sepsis = pd.read_csv('data/FinalNonSepsisCohort.csv')
    df_non_sepsis = df_non_sepsis.set_index('PatientID').drop('deathperiod',
                                                              axis='columns')
    df_sepsis = df_sepsis.set_index('PatientID').drop('deathperiod',
                                                      axis='columns')
    return df_sepsis, df_non_sepsis


if __name__ == '__main__':
    df_sepsis, df_non_sepsis = read_prepare_static_data()
    df_ts_pred = fit_predict_time_series()

    # (X_train, X_test, y_train,
    #  y_test), X, y = get_train_test_time_series_dataset(
    #     df_non_sepsis,
    #     df_sepsis,
    #     df_ts_pred)

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    print(model.score(X_test, y_test))
