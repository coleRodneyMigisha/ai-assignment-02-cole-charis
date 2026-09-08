EDA and preprocessing for Known_Patients_01.csv

Run the analysis script which produces plots and prints recommendations:

```bash
python eda_known_patients.py
```

Outputs:
- `plots/` directory with PNG figures (distributions, boxplots, heatmap, pairplot, learning curves)

Notes:
- Script will infer the target column (looks for common names, otherwise picks last column).
- Categorical encoding: One-hot for low-cardinality, frequency encoding for high-cardinality.

## Supervised classification

`Known_Patients_01.csv` is labelled with the binary target `cerebrovascular_accident`:
`0` means no cerebrovascular accident and `1` means a cerebrovascular accident. This
is therefore a supervised binary classification task, not a continuous prediction
(regression) task. `New_Patients_01.csv` contains patients whose target values are
unknown and must be predicted.

Run the modular model comparison with:

```bash
python train_classifiers.py
```

The script compares Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM, and
Random Forest. It uses imputation, scaling, and one-hot encoding inside a scikit-learn
pipeline to prevent preprocessing leakage. Results include accuracy, precision, recall,
F1, and ROC AUC; the best model is selected by holdout F1 and writes predictions to
`New_Patients_predictions.csv`.

## Advanced classifiers and imbalance

Run the additional experiments with:

```bash
python train_advanced_classifiers.py
```

This module adds AdaBoost and XGBoost. It compares XGBoost class weighting with
SMOTE and evaluates thresholds of `0.50` and `0.35`. Lowering the threshold usually
increases recall while reducing precision; SMOTE can improve minority recall, but
must be judged using precision, recall, ROC AUC, and log loss rather than accuracy
alone. The module also tunes XGBoost tree depth, child weight, learning rate,
sampling, and L1/L2 regularization with Optuna. Finally, it evaluates LightGBM with
custom focal loss using ROC AUC and log loss.

### Accuracy and metric interpretation

The advanced workflow compares baseline and adjusted hyperparameters for both
AdaBoost and XGBoost. It writes the following files to `plots/advanced_metrics/`:

- `metrics.csv`: all numeric results for each model.
- `confusion_matrices.png`: true negatives, false positives, false negatives,
	and true positives.
- `roc_curves.png`: recall versus false-positive rate across probability thresholds.
- `metric_comparison.png`: accuracy, precision, recall, F1, and ROC AUC comparison.

Metric implications:

- **Accuracy** is the proportion of all correct predictions. It can be misleading
	for imbalanced stroke data because predicting the majority class can look good.
- **Precision** is the proportion of predicted positive cases that are truly positive.
	Higher precision means fewer false alarms.
- **Recall** is the proportion of actual positive cases detected. Higher recall means
	fewer missed positive patients, which is often important in medical screening.
- **F1** is the harmonic mean of precision and recall, useful when both error types
	matter.
- **ROC AUC** measures ranking quality across all thresholds; `0.5` is random and
	`1.0` is perfect separation.
- **Log loss** penalizes incorrect probabilities, especially confident wrong ones.

For the medical use case, select a threshold based on the cost of false negatives,
then report the resulting precision and recall alongside ROC AUC and log loss.

## Evaluate all models

Run the combined evaluator to compare Group A and Group B models on the same
stratified holdout set:

```bash
python evaluate_models.py
```

The evaluator writes the complete metrics table to
`reports/all_model_evaluation.md` and saves confusion matrices, ROC curves,
classification metric plots, log-loss plots, and CSV results under
`plots/all_model_evaluation/`. MAE and RMSE are intentionally excluded because
this is a classification task.

## Standalone data preparation

Run the dedicated preparation script before modelling:

```bash
python data_preparation.py
```

It reads `Known_Patients_01.csv` and `New_Patients_01.csv`, removes identifier
columns, validates the binary target, imputes missing numeric values with the
training median, imputes missing categorical values with the most frequent value,
one-hot encodes nominal categories, and standardises numeric features. The
transformer is fitted on known patients only and then applied to new patients to
avoid data leakage.

Prepared files are written to `prepared_data/`:

- `known_prepared.csv`
- `new_prepared.csv`
- `preparation_metadata.json`

## Complete workflow

Run the assignment workflow from the repository root:

```bash
python main.py
```

This generates EDA plots, Group A and Group B metric reports, predictions for the
new patients, and XAI outputs under `plots/xai/`. The aggregate advanced report is
`reports/advanced_metrics.csv`; optional Optuna, SMOTE, LightGBM, SHAP, and LIME
experiments print an install message when their packages are unavailable. Evidently
drift configuration is intentionally reserved for the project owner.

Install the optional packages first:

```bash
pip install -r requirements.txt
```
