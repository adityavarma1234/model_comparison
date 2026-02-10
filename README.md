# German Credit Risk Classification

## Problem Statement
The goal of this project is to predict whether a bank customer represents a good or bad credit risk based on attributes such as checking status, duration, credit history, and employment.

## Dataset Description
**Source:** UCI Machine Learning Repository (German Credit Data)
**Attributes:** 20 Features (7 numerical, 13 categorical)
**Target:** Good (0) or Bad (1) Credit Risk
**Size:** 1000 Instances

### Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.755 | 0.791 | 0.618 | 0.548 | 0.581 | 0.412 |
| **Decision Tree** | 0.695 | 0.652 | 0.492 | 0.516 | 0.504 | 0.265 |
| **KNN** | 0.730 | 0.745 | 0.565 | 0.484 | 0.521 | 0.342 |
| **Naive Bayes** | 0.740 | 0.782 | 0.574 | 0.629 | 0.600 | 0.405 |
| **Random Forest (Ensemble)** | 0.775 | 0.812 | 0.655 | 0.565 | 0.607 | 0.461 |
| **XGBoost (Ensemble)** | 0.780 | 0.818 | 0.648 | 0.613 | 0.630 | 0.478 |

### Model Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Provided a solid baseline with the second-highest AUC score. It handled the binary classification task well but struggled slightly with non-linear relationships compared to ensemble methods. |
| **Decision Tree** | This model had the lowest accuracy and MCC score. It likely overfitted the training data, failing to generalize well to the unseen test set compared to the Random Forest. |
| **KNN** | Performance was moderate (Accuracy: ~73%). It was highly sensitive to the high dimensionality (20 features) and required strict feature scaling to function correctly. |
| **Naive Bayes** | Performed surprisingly well (Recall: ~63%), likely because many of the categorical features (like checking status) align well with probability-based predictions. |
| **Random Forest (Ensemble)** | One of the best performers. By averaging multiple trees, it successfully reduced the variance seen in the single Decision Tree model and handled the mix of categorical and numerical data effectively. |
| **XGBoost (Ensemble)** | The overall best model for this dataset. It effectively minimized the logloss function and captured complex interactions between features like 'Duration' and 'Credit Amount'. |

## Stream-lit app ur
* https://modelcomparison-nqmqmzuiubluns5flqwzb2.streamlit.app/