import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from time_series.sktime_experiments import \
    fit_predict_time_series_separate_classification
from utils import merge_static_series_pred
from sklearn.utils import class_weight
import matplotlib.pyplot as plt


def read_prepare_static_data():
    df_static_sepsis = pd.read_csv('data/FinalSepsisCohort.csv')
    df_static_non_sepsis = pd.read_csv('data/FinalNonSepsisCohort.csv')

    df_static_non_sepsis = df_static_non_sepsis.drop('deathperiod',
                                                     axis='columns')
    df_static_sepsis = df_static_sepsis.drop('deathperiod',
                                             axis='columns')

    # exclude patients from non sepsis if they are in sepsis file
    df_static_non_sepsis = df_static_non_sepsis[
        ~df_static_non_sepsis.index.isin(df_static_sepsis.index.values)]

    return df_static_non_sepsis, df_static_sepsis


def get_xgboost_X_enhanced():
    df_static_sepsis, df_static_non_sepsis = read_prepare_static_data()
    df_ts_pred, X_series, _ = \
        fit_predict_time_series_separate_classification(
            './data/preprocessed_data/series_non_sepsis.pkl',
            './data/preprocessed_data/series_sepsis.pkl')
    X, y = merge_static_series_pred(df_static_non_sepsis,
                                    df_static_sepsis,
                                    df_ts_pred)

    encoders = []
    for column_name in df_ts_pred.columns[1:]:
        le = LabelEncoder()
        X[column_name] = le.fit_transform(X[column_name])
        encoders.append(le)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2137)
    # model = xgb.XGBClassifier()

    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    param = {'max_depth': X.columns.shape[0], 'objective': 'binary:logistic',
             'nthread': 4}

    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train, sample_weight=classes_weights)

    # le = LabelEncoder()
    # dtrain = xgb.DMatrix(X_train, label=le.fit_transform(y_train), weight=classes_weights)
    #
    # model = xgb.train(param, dtrain)

    # model.fit(X_train, y_train, sample_weight=classes_weights)
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


def plot_tree(model):
    xgb.plot_tree(model)
    plt.show()


if __name__ == '__main__':
    model, X, X_display, y = get_xgboost_X_enhanced()
