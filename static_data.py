import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from time_series.sktime_experiment import fit_predict_time_series_column_ensemble
from utils import merge_static_series_pred


def read_prepare_static_data():
    df_static_sepsis = pd.read_csv('data/FinalSepsisCohort.csv')
    df_static_non_sepsis = pd.read_csv('data/FinalNonSepsisCohort.csv')

    df_static_non_sepsis = df_static_non_sepsis.drop('deathperiod',
                                                     axis='columns')
    df_static_sepsis = df_static_sepsis.drop('deathperiod',
                                             axis='columns')

    df_static_non_sepsis = df_static_non_sepsis[
        ~df_static_non_sepsis.index.isin(df_static_sepsis.index.values)]

    return df_static_non_sepsis, df_static_sepsis


def get_xgboost_X_enhanced():
    df_static_sepsis, df_static_non_sepsis = read_prepare_static_data()
    df_ts_pred, X_series = fit_predict_time_series_column_ensemble()
    X, y = merge_static_series_pred(df_static_non_sepsis,
                                    df_static_sepsis,
                                    df_ts_pred)

    le = LabelEncoder()
    X['TSPred'] = le.fit_transform(X['TSPred'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2137)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    X_display = X.__deepcopy__()
    X_display['TSPred'] = le.inverse_transform(X_display['TSPred'])
    return model, X, X_display


if __name__ == '__main__':
    get_xgboost_X_enhanced()
