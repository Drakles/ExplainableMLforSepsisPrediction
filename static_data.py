import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from time_series.sktime_experiments import \
    fit_predict_time_series_column_ensemble, \
    fit_predict_time_series_separate_classification
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
    # df_ts_pred, X_series = fit_predict_time_series_column_ensemble()
    df_ts_pred, X_series = fit_predict_time_series_separate_classification()
    X, y = merge_static_series_pred(df_static_non_sepsis,
                                    df_static_sepsis,
                                    df_ts_pred)

    encoders = []
    for column_name in df_ts_pred.columns[1:]:
        le = LabelEncoder()
        X[column_name] = le.fit_transform(X[column_name])
        encoders.append(le)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2137)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    print('mean accuracy: ' + str(model.score(X_test, y_test)))
    print('f1 score: ' + str(f1_score(y_test, model.predict(X_test),
                                      average='weighted')))

    # build X_display
    X_display = X.__deepcopy__()
    for i in range(len(df_ts_pred.columns) - 1):
        column_name = df_ts_pred.columns[i + 1]
        le = encoders[i]
        X_display[column_name] = le.inverse_transform(X[column_name])

    return model, X, X_display, y


if __name__ == '__main__':
    get_xgboost_X_enhanced()
