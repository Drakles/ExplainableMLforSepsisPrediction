import numpy as np
import pylab as pl
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence

from static_data import get_xgboost_X_enhanced


def summary_plot(shap_values):
    shap.summary_plot(shap_values, X,
                      max_display=X.shape[1])


def features_importance_bar(shap_values, X, limit):
    shap.summary_plot(shap_values=shap_values,
                      feature_names=X.columns.values, plot_type='bar',
                      max_display=limit, plot_size=(18, 18))


def dependency_plot_by_feature(column_name, X):
    column_index = X.columns.get_loc(column_name)
    shap.dependence_plot(column_index, explainer.shap_values(X), X)


def beeswarm_plot(shap_values, max_display):
    shap.plots.beeswarm(shap_values, max_display=max_display,
                        plot_size=(24, 20))


def scatter_dependence_plot(feature_name):
    shap.plots.scatter(shap_values[:, feature_name],x_jitter=0.5)


def scatter_dependence_with_interaction_with_other(shap_values, feature_name,
                                                   other_feature_name):
    shap.plots.scatter(shap_values[:, feature_name], color=shap_values[:,
                                                           other_feature_name])


def dependence_plot(feature_name, X):
    shap.dependence_plot(feature_name, explainer.shap_values(X), X,
                         interaction_index=None)


def interaction_value_between_features(features_tuple,
                                       shap_interaction_values, X):
    shap.dependence_plot(
        features_tuple,
        shap_interaction_values, X,
    )


def plot_matrix(shap_interaction_values, X, limit):
    tmp = np.abs(shap_interaction_values).sum(0)
    for i in range(tmp.shape[0]):
        tmp[i, i] = 0
    inds = np.argsort(-tmp.sum(0))[:limit]
    tmp2 = tmp[inds, :][:, inds]
    pl.figure(figsize=(12, 12))
    pl.imshow(tmp2)
    pl.yticks(range(tmp2.shape[0]), X.columns[inds], rotation=50.4,
              horizontalalignment="right")
    pl.xticks(range(tmp2.shape[0]), X.columns[inds], rotation=50.4,
              horizontalalignment="left")
    pl.gca().xaxis.tick_top()
    pl.show()


def plot_waterfall(shap_values, index):
    shap.plots.waterfall(shap_values[index])


def single_force_plot(explainer, shap_values, X, index):
    shap.force_plot(explainer.expected_value, shap_values.values[index],
                    X.iloc[index],
                    matplotlib=True)


def partial_dependence(model, X, column_name, kind='both'):
    column_index = X.columns.get_loc(column_name)
    plot_partial_dependence(model, X, [column_index], kind=kind)
    plt.show()


if __name__ == '__main__':
    model, X, y = get_xgboost_X_enhanced()

    explainer = shap.TreeExplainer(model)
    # shap_interaction_values = explainer.shap_interaction_values(X)
    # print('calculating shap interaction values completed')
    #
    # features_importance_bar(explainer(X), X, 20)
    # beeswarm_plot(explainer(X), 11)
