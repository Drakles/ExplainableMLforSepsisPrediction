import shap
from sktime_experiment import get_train_test_dataset, column_concatenate_clf

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_train_test_dataset()
    clf = column_concatenate_clf(X_train, y_train)

    explainer = shap.KernelExplainer(clf.predict_proba,X_train,link="logit")

    shap_values = explainer.shap_values(X_test, nsamples=100)

    # plot the SHAP values for the Setosa output of the first instance
    shap.force_plot(explainer.expected_value[0], shap_values[0][0, :],
                    X_test.iloc[0, :], link="logit")
