# Patient Stroke Classification Presentation

Maximum: 12 slides.

1. **Problem and objective**: classify cerebrovascular accident risk for known and new patients.
2. **Data**: 4,644 labelled patients, 466 new patients, features, missing BMI values.
3. **Target and imbalance**: binary target, 232 positives and 4,412 negatives.
4. **EDA findings**: distributions, categorical counts, correlations, outliers, and classification bias/variance.
5. **Preprocessing**: train-only imputation, scaling, one-hot encoding, and ID exclusion.
6. **Group A models**: Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM, Random Forest.
7. **Group A results**: accuracy, precision, recall, F1, ROC AUC, confusion matrix, and selected model.
8. **Group B models**: AdaBoost and XGBoost with adjusted hyperparameters.
9. **Imbalance experiments**: class weighting, SMOTE, thresholds, precision/recall trade-off.
10. **Tuning and focal loss**: Optuna XGBoost parameters and LightGBM focal-loss ROC AUC/log loss.
11. **Explainability**: SHAP global importance, LIME local explanation, and counterfactual examples.
12. **Conclusion and limitations**: threshold choice, no labels for new patients, and Evidently drift analysis to be configured.