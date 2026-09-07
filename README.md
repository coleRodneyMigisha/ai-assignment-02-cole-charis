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

Install the optional packages first:

```bash
pip install -r requirements.txt
```
