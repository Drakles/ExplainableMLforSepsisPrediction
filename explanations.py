import numpy as np
import pylab as pl
import shap


def get_interaction_values(explainer, X):
    return explainer.shap_interaction_values(X)


def summary_plot(shap_interaction_values, X):
    shap.summary_plot(shap_interaction_values, X)


def dependency_plot_by_feature(feature, shap_values, X):
    shap.dependence_plot(feature, shap_values, X)


def beeswarm_plot(shap_values):
    shap.plots.beeswarm(shap_values)


def plot_matrix(shap_interaction_values, X):
    tmp = np.abs(shap_interaction_values).sum(0)
    for i in range(tmp.shape[0]):
        tmp[i, i] = 0
    inds = np.argsort(-tmp.sum(0))[:50]
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
    shap.force_plot(explainer.expected_value, shap_values[index], X.iloc[index],
                    matplotlib=True)


if __name__ == '__main__':
    model, X = None, None

    explainer = shap.TreeExplainer(model, data=X,
                                   model_output="probability")
    shap_values = explainer.shap_values(X)
