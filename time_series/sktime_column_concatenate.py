import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.transformers.series_as_features.compose import ColumnConcatenator

from utils import get_train_test_time_series_dataset

if __name__ == '__main__':
    series_non_sepsis_df = pd.read_pickle(
        '../data/preprocessed_data/union_features/series_non_sepsis.pkl')
    series_sepsis_df = pd.read_pickle(
        '../data/preprocessed_data/union_features/series_sepsis.pkl')

    (X_train, X_test, y_train,
     y_test), X, y = get_train_test_time_series_dataset(
        series_non_sepsis_df,
        series_sepsis_df)

    steps = [
        ("concatenate", ColumnConcatenator()),
        ("classify", TimeSeriesForestClassifier(class_weight='balanced',
                                                verbose=True,
                                                n_jobs=-1)),
    ]
    model = Pipeline(steps, verbose=True)
    model.fit(X_train, y_train)

    print('f1 score: ' + str(f1_score(y_test, model.predict(X_test),
                                      average='weighted')))
    predictions = model.predict_proba(X_test)
    print('roc auc: ' + str(roc_auc_score(y_test, predictions[:, 1])))

    # f1 score: 0.9332825436660429
    # roc auc: 0.9859991039426523
