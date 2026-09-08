"""Evaluate every available classifier and create one combined report.

The evaluation set is a single stratified holdout shared by all models. The
reported metrics are classification metrics only: accuracy, precision, recall,
F1, ROC AUC, and log loss.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from train_classifiers import (
    RANDOM_STATE,
    build_models,
    build_preprocessor,
    calculate_metrics,
    load_datasets,
    prepare_features,
)
from train_advanced_classifiers import (
    build_adaboost,
    build_tuned_adaboost,
    build_tuned_xgboost,
    build_xgboost,
    transform_data,
)


OUTPUT_DIR = Path("plots/all_model_evaluation")
REPORT_PATH = Path("reports/all_model_evaluation.md")


def evaluate_pipeline_models(X_train, y_train, X_test, y_test):
    """Fit raw-data pipelines for Group A models and collect their results."""
    results = {}
    for name, estimator in build_models().items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(X_train)), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        results[name] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": calculate_metrics(y_test, predictions, probabilities),
        }
    return results


def evaluate_advanced_models(X_train, y_train, X_test, y_test):
    """Fit Group B estimators on the same transformed holdout data."""
    train_matrix, test_matrix, _ = transform_data(X_train, X_test, X_test)
    estimators = {
        "AdaBoost": build_adaboost(),
        "AdaBoost adjusted": build_tuned_adaboost(),
    }
    optional_estimators = {
        "XGBoost": build_xgboost,
        "XGBoost adjusted": build_tuned_xgboost,
    }
    for name, builder in optional_estimators.items():
        try:
            estimators[name] = builder()
        except ImportError as error:
            print(f"Skipping {name}: {error}")

    results = {}
    for name, estimator in estimators.items():
        estimator.fit(train_matrix, y_train)
        predictions = estimator.predict(test_matrix)
        probabilities = estimator.predict_proba(test_matrix)[:, 1]
        results[name] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": calculate_metrics(y_test, predictions, probabilities),
        }
    return results


def save_plots(results, y_test, output_dir=OUTPUT_DIR):
    """Save confusion matrices, ROC curves, and all metric comparisons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(results)

    columns = 3
    rows = (len(model_names) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for axis, name in zip(axes, model_names):
        matrix = confusion_matrix(y_test, results[name]["predictions"])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
    for axis in axes[len(model_names):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrices_all_models.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 7))
    for name in model_names:
        RocCurveDisplay.from_predictions(
            y_test,
            results[name]["probabilities"],
            name=name,
            ax=axis,
        )
    axis.set_title("ROC curves for all classifiers")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curves_all_models.png", dpi=150)
    plt.close(figure)

    metrics = pd.DataFrame(
        {name: values["metrics"] for name, values in results.items()}
    ).T
    metrics.to_csv(output_dir / "all_model_metrics.csv")

    metrics[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot.bar(
        figsize=(14, 7), ylim=(0, 1)
    )
    plt.title("Classification metrics for all models")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "classification_metrics_all_models.png", dpi=150)
    plt.close()

    metrics[["log_loss"]].plot.bar(figsize=(10, 6))
    plt.title("Log loss for all models")
    plt.ylabel("Log loss, lower is better")
    plt.xlabel("Model")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "log_loss_all_models.png", dpi=150)
    plt.close()
    return metrics


def write_report(metrics, path=REPORT_PATH):
    """Write the complete all-model evaluation as Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rounded = metrics.reset_index().rename(columns={"index": "model"}).round(4)
    try:
        table = rounded.to_markdown(index=False)
    except ImportError:
        headers = "| " + " | ".join(rounded.columns) + " |"
        separator = "| " + " | ".join("---" for _ in rounded.columns) + " |"
        rows = [
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in rounded.itertuples(index=False, name=None)
        ]
        table = "\n".join([headers, separator, *rows])

    best_f1 = metrics["f1"].idxmax()
    best_auc = metrics["roc_auc"].idxmax()
    report = f"""# All-Model Evaluation

All models use the same stratified 20% holdout set. The target is a highly
imbalanced binary classification target, so no single metric should be used
alone.

## Metrics

{table}

## Findings

- Highest F1 score: **{best_f1}**.
- Highest ROC AUC: **{best_auc}**.
- Accuracy may be inflated by the majority class.
- Precision measures false-alarm control; recall measures missed-positive control.
- F1 balances precision and recall.
- ROC AUC measures ranking quality across thresholds.
- Log loss measures the quality and confidence of predicted probabilities.

## Visualizations

- `plots/all_model_evaluation/confusion_matrices_all_models.png`
- `plots/all_model_evaluation/roc_curves_all_models.png`
- `plots/all_model_evaluation/classification_metrics_all_models.png`
- `plots/all_model_evaluation/log_loss_all_models.png`
"""
    path.write_text(report, encoding="utf-8")


def main():
    known, new = load_datasets()
    X, y, _ = prepare_features(known, new)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    results = evaluate_pipeline_models(X_train, y_train, X_test, y_test)
    results.update(evaluate_advanced_models(X_train, y_train, X_test, y_test))
    metrics = save_plots(results, y_test)
    write_report(metrics)
    print(metrics.round(4).to_string())
    print(f"\nPlots written to {OUTPUT_DIR}/")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
