# SepsisDatasetAnalyse
This code was part of the final year project for Explainable Machine Learning for Sepsis Prediction. 


The data used to solve the sepsis classification problem needed to be cleaned to remove irrelevant features preprocessed, which included data interpolation, imputation, and transformation in the case of time series features.
In the case of static features, most of the effort related to data preprocessing was related to the reconstruction of the age distribution.
The problem of sepsis detection involves time-series features, and this type of data is not compatible with current explainability frameworks.
To address this problem, custom architecture was proposed, which can be adjusted to include any other model to handle dynamic and static data.
Additionally, the proposed architecture can group explicitly related time series features according to the user’s need.
Many different techniques were involved in contradicting the imbalance data as they may have a potentially harmful impact when training or evaluating the model after completing the training, including stratified k cross-validation, weighted f1 score, and computing sample weights.
Even though this project’s main goal is related to explainability and not the model’s performance, basic fine-tuning of the hyperparameters was performed using grid search cross-validation.
Model predictions was validated with the use of SHAP framework.
As a result, some of the features, which potentially were correlated with external factors not included in the dataset and which could not be explained based on the available data and domain knowledge, were removed from the dataset.
Then the model was again evaluated. As a result, we have achieved a more robust model with more predictable behaviour, with only slightly worse prediction score when compared with the model trained on the original set of features.
