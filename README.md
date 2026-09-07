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
- **MAE** and **RMSE** are included as probability-error diagnostics against binary
	labels. They are not substitutes for classification metrics. RMSE penalizes large
	probability errors more heavily than MAE.

For the medical use case, select a threshold based on the cost of false negatives,
then report the resulting precision and recall alongside ROC AUC and log loss.

Install the optional packages first:

```bash
pip install -r requirements.txt
```
