import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sktime.classification.compose import TimeSeriesForestClassifier, \
    ColumnEnsembleClassifier

from time_series.sktime_column_ensemble import get_estimators


def fit_predict_time_series_hybrid_classification(sepsis_path, non_sepsis_path):
    series_non_sepsis_df = pd.read_pickle(non_sepsis_path)
    series_non_sepsis_df['Label'] = np.array(['non_sepsis' for _ in range(
        len(series_non_sepsis_df))])
    series_sepsis_df = pd.read_pickle(sepsis_path)
    series_sepsis_df['Label'] = np.array(['sepsis' for _ in range(
        len(series_sepsis_df))])

    X = series_non_sepsis_df.append(series_sepsis_df)
    y = X['Label']
    X = X.drop('Label', axis=1)

    X_nbp = X[['NBP Mean', 'NBP [Diastolic]', 'NBP [Systolic]']]
    X_o2 = X[['SaO2', 'SpO2']]

    X_one_clf_per_feature = X.drop(['Arterial PaCO2', 'Arterial PaO2'] +
                                   ['NBP Mean',
                                    'NBP [Diastolic]',
                                    'NBP [Systolic]'] + [
                                       'SaO2', 'SpO2'], axis=1)

    predictions_per_feature = {}
    predictions_per_feature['PatientID'] = np.array(X.index, dtype=int)

    # nbp
    update_predictions_per_feature_group(X_nbp, 'NBP-TSP',
                                         predictions_per_feature, y)

    # o2
    update_predictions_per_feature_group(X_o2, 'O2-TSP',
                                         predictions_per_feature, y)

    for f_index in range(len(X_one_clf_per_feature.columns)):
        X_one_column = pd.DataFrame(X_one_clf_per_feature.iloc[:, f_index])

        feature_name = str(X_one_clf_per_feature.columns[f_index]) \
                           .replace('[', '-') \
                           .replace(']', '') \
                       + '-TSP'
        model = TimeSeriesForestClassifier(n_estimators=5,
                                           class_weight='balanced')

        predictions_per_feature[feature_name] = \
            cross_val_predict(model, X_one_column, y, cv=StratifiedKFold(),
                              method='predict_proba').T[1]

    return pd.DataFrame(data=predictions_per_feature)


def update_predictions_per_feature_group(X_feature_grouped, feature_group_name,
                                         predictions_per_feature, y):
    model = Pipeline([('classify', ColumnEnsembleClassifier(get_estimators(
        len(X_feature_grouped.columns))),)])

    # scores = cross_validate(model, X_feature_grouped, y,
    #                         scoring=['f1_weighted', 'roc_auc'], verbose=1,
    #                         cv=StratifiedKFold(shuffle=False))
    # print(scores)

    predictions_per_feature[feature_group_name] = \
        cross_val_predict(model, X_feature_grouped, y, cv=StratifiedKFold(),
                          method='predict_proba').T[1]


if __name__ == '__main__':
    df_pred = fit_predict_time_series_hybrid_classification(
        '../data/preprocessed_data/union_features/series_sepsis.pkl',
        '../data/preprocessed_data/union_features/series_non_sepsis.pkl'
    )
