"""Train and compare supervised classifiers for the patient datasets.

The known-patient file contains the binary target ``cerebrovascular_accident``.
The new-patient file is used only after model comparison to generate predictions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


TARGET_COLUMN = "cerebrovascular_accident"
ID_COLUMNS = ["id"]
RANDOM_STATE = 42
METRICS_PLOT_DIR = Path("plots/classifier_metrics")
METRICS_REPORT_PATH = Path("reports/classifier_evaluation.md")


def load_datasets(
    known_path: str | Path = "Known_Patients_01.csv",
    new_path: str | Path = "New_Patients_01.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the labelled training data and unlabelled prediction data."""
    return pd.read_csv(known_path), pd.read_csv(new_path)


def prepare_features(
    known: pd.DataFrame,
    new: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Separate the binary target and align training/prediction features."""
    if target_column not in known.columns:
        raise ValueError(f"Target column '{target_column}' was not found in known data.")

    labelled = known.dropna(subset=[target_column]).copy()
    target = labelled.pop(target_column)
    if target.nunique() != 2:
        raise ValueError("This classifier workflow requires exactly two target classes.")

    feature_columns = [column for column in labelled.columns if column not in ID_COLUMNS]
    missing_features = set(feature_columns) - set(new.columns)
    if missing_features:
        raise ValueError(f"New data is missing feature columns: {sorted(missing_features)}")

    return labelled[feature_columns], target, new[feature_columns].copy()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create numeric and categorical preprocessing without data leakage."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)


def build_decision_tree() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)


def build_knn() -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=5)


def build_naive_bayes() -> GaussianNB:
    return GaussianNB()


def build_svm() -> SVC:
    return SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_models() -> dict[str, object]:
    """Return each requested Group A classifier from its individual builder."""
    return {
        "Logistic Regression": build_logistic_regression(),
        "Decision Tree": build_decision_tree(),
        "KNN": build_knn(),
        "Naive Bayes": build_naive_bayes(),
        "SVM": build_svm(),
        "Random Forest": build_random_forest(),
    }


def calculate_metrics(y_true, predictions, probabilities) -> dict[str, float]:
    """Calculate classification and probability-error metrics."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
    }


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Evaluate each model with a stratified holdout and five-fold CV."""
    models = models or build_models()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    results = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor(X_train)), ("model", model)]
        )
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=5, scoring=scoring)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, predictions, probabilities)
        results.append(
            {
                "model": name,
                "cv_f1_mean": cv_scores["test_f1"].mean(),
                "cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
        fitted_pipelines[name] = pipeline

    return pd.DataFrame(results).sort_values("test_f1", ascending=False), fitted_pipelines


def save_evaluation_plots(
    evaluation_data: dict[str, dict],
    output_dir: Path = METRICS_PLOT_DIR,
) -> None:
    """Save one confusion matrix per model plus shared metric plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(evaluation_data)
    metrics = pd.DataFrame(
        {name: details["metrics"] for name, details in evaluation_data.items()}
    ).T

    for name, details in evaluation_data.items():
        matrix = confusion_matrix(details["y_test"], details["predictions"])
        figure, axis = plt.subplots(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(f"{name} confusion matrix")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        figure.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        figure.savefig(output_dir / f"confusion_matrix_{safe_name}.png", dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for name, details in evaluation_data.items():
        from sklearn.metrics import RocCurveDisplay

        RocCurveDisplay.from_predictions(
            details["y_test"], details["probabilities"], name=name, ax=axis
        )
    axis.set_title("ROC curves: Group A classifiers")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curves_all_models.png", dpi=150)
    plt.close(figure)

    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metrics[metric_columns].plot.bar(figsize=(12, 6), ylim=(0, 1))
    plt.title("Classification metrics by model")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "classification_metrics_all_models.png", dpi=150)
    plt.close()

    metrics[["log_loss"]].plot.bar(figsize=(8, 6))
    plt.title("Log loss by model")
    plt.ylabel("Log loss (lower is better)")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "log_loss_all_models.png", dpi=150)
    plt.close()
    metrics.to_csv(output_dir / "all_model_metrics.csv")


def evaluate_models_individually(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Evaluate every model and retain predictions for model-specific plots."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    evaluation_data = {}
    for name, model in build_models().items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor(X_train)), ("model", model)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        evaluation_data[name] = {
            "model": pipeline,
            "y_test": y_test,
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": calculate_metrics(y_test, predictions, probabilities),
        }
    rows = [dict(model=name, **details["metrics"]) for name, details in evaluation_data.items()]
    return pd.DataFrame(rows).sort_values("f1", ascending=False), evaluation_data


def write_evaluation_report(results: pd.DataFrame, path: Path = METRICS_REPORT_PATH) -> None:
    """Write all Group A metrics and their interpretation to Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rounded_results = results.round(4)
    try:
        table = rounded_results.to_markdown(index=False)
    except ImportError:
        # Keep report generation working when the optional tabulate package is absent.
        headers = "| " + " | ".join(rounded_results.columns) + " |"
        separator = "| " + " | ".join("---" for _ in rounded_results.columns) + " |"
        rows = [
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in rounded_results.itertuples(index=False, name=None)
        ]
        table = "\n".join([headers, separator, *rows])
    report = f"""# Group A Classifier Evaluation

The models were evaluated on the same stratified 20% holdout set. The target is
binary and imbalanced, so accuracy is reported together with precision, recall,
F1, ROC AUC, and log loss.

## Results

{table}

## Metric interpretation

- **Accuracy**: fraction of all predictions that are correct; can be misleading when the negative class dominates.
- **Precision**: fraction of predicted positive patients who are truly positive; higher precision means fewer false alarms.
- **Recall**: fraction of actual positive patients detected; higher recall means fewer missed cases.
- **F1**: harmonic mean of precision and recall; useful when both error types matter.
- **ROC AUC**: threshold-independent ranking quality; 0.5 is random and 1.0 is perfect.
- **Log loss**: probability quality; confident incorrect probabilities receive a large penalty.

## Plots

- `plots/classifier_metrics/roc_curves_all_models.png`
- `plots/classifier_metrics/classification_metrics_all_models.png`
- `plots/classifier_metrics/log_loss_all_models.png`
- Individual confusion matrices are saved in the same directory.
"""
    path.write_text(report, encoding="utf-8")


def fit_best_model(
    model_name: str,
    models: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
) -> Pipeline:
    """Fit the selected model on all known labelled patients."""
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models)}")
    pipeline = Pipeline(
        steps=[("preprocessor", build_preprocessor(X)), ("model", models[model_name])]
    )
    return pipeline.fit(X, y)


def predict_new_patients(model: Pipeline, X_new: pd.DataFrame) -> pd.Series:
    """Generate target predictions for new patients."""
    return pd.Series(model.predict(X_new), index=X_new.index, name=TARGET_COLUMN)


def add_predictions(
    new_data: pd.DataFrame,
    predictions: pd.Series,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Add predictions while preserving the target spelling in the new file."""
    output = new_data.copy()
    output_target = next(
        (column for column in output.columns if column.strip().lower() == target_column.replace("_", " ")),
        target_column,
    )
    output[output_target] = predictions.to_numpy()
    return output


def main() -> None:
    known, new = load_datasets()
    X, y, X_new = prepare_features(known, new)
    models = build_models()
    results, _ = evaluate_models(X, y, models)
    individual_results, evaluation_data = evaluate_models_individually(X, y)
    save_evaluation_plots(evaluation_data)
    write_evaluation_report(individual_results)

    print("Task: binary supervised classification")
    print(f"Target: {TARGET_COLUMN} (classes: {sorted(y.unique().tolist())})")
    print("\nModel comparison including cross-validation and holdout metrics:")
    print(results.round(3).to_string(index=False))
    print("\nComplete individual evaluation metrics:")
    print(individual_results.round(3).to_string(index=False))
    print(f"\nPlots written to {METRICS_PLOT_DIR}/")
    print(f"Markdown report written to {METRICS_REPORT_PATH}")

    best_model_name = individual_results.iloc[0]["model"]
    best_model = fit_best_model(best_model_name, models, X, y)
    predictions = predict_new_patients(best_model, X_new)
    output = add_predictions(new, predictions)
    output.to_csv("New_Patients_predictions.csv", index=False)
    print(f"\nSelected model: {best_model_name}")
    print("Predictions written to New_Patients_predictions.csv")


if __name__ == "__main__":
    main()