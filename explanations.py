import numpy as np
import pylab as pl
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import plot_partial_dependence

from static_data import get_xgboost_X_enhanced


# notebook for dependency plot:
# https://slundberg.github.io/shap/notebooks/plots/dependence_plot.html

# https://slundberg.github.io/shap/notebooks/League%20of%20Legends%20Win
# %20Prediction%20with%20XGBoost.html

# https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html


def summary_plot(shap_interaction_values, X_display):
    shap.summary_plot(shap_interaction_values, X_display,
                      max_display=X.shape[1])


def features_interaction_bar(shap_values, X, X_display):
    shap.summary_plot(shap_values=shap_values, features=X,
                      feature_names=X_display.columns.values, plot_type='bar',
                      max_display=X.shape[1])


def dependency_plot_by_feature(column_name, X):
    column_index = X.columns.get_loc(column_name)
    shap.dependence_plot(column_index, explainer.shap_values(X), X)


def beeswarm_plot(shap_values):
    shap.plots.beeswarm(shap_values,max_display=15)


def plot_matrix(shap_interaction_values, X_display):
    tmp = np.abs(shap_interaction_values).sum(0)
    for i in range(tmp.shape[0]):
        tmp[i, i] = 0
    inds = np.argsort(-tmp.sum(0))[:50]
    tmp2 = tmp[inds, :][:, inds]
    pl.figure(figsize=(12, 12))
    pl.imshow(tmp2)
    pl.yticks(range(tmp2.shape[0]), X_display.columns[inds], rotation=50.4,
              horizontalalignment="right")
    pl.xticks(range(tmp2.shape[0]), X_display.columns[inds], rotation=50.4,
              horizontalalignment="left")
    pl.gca().xaxis.tick_top()
    pl.show()


def plot_waterfall(shap_values, index):
    shap.plots.waterfall(shap_values[index])


def single_force_plot(explainer, shap_values, X_display, index):
    shap.force_plot(explainer.expected_value, shap_values.values[index],
                    X_display.iloc[index],
                    matplotlib=True)


def partial_dependence(model, X, column_name, kind='both'):
    column_index = X.columns.get_loc(column_name)
    plot_partial_dependence(model, X, [column_index], kind=kind)
    plt.show()


if __name__ == '__main__':
    model, X, X_display, y = get_xgboost_X_enhanced()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap_interaction_values = explainer.shap_interaction_values(X)
