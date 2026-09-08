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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
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
RANDOM_SEARCH_ITERATIONS = 12
RANDOM_SEARCH_CV = 3


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


def _collect_results(estimators, X_train, y_train, X_test, y_test):
    """Fit estimators on matrices and return predictions plus metrics."""
    results = {}
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        results[name] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": calculate_metrics(y_test, predictions, probabilities),
        }
    return results


def group_a_search_spaces():
    """Return reproducible random-search spaces for each Group A model."""
    return {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__solver": ["liblinear", "lbfgs"],
        },
        "Decision Tree": {
            "model__max_depth": [3, 5, 8, 12, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__criterion": ["gini", "entropy", "log_loss"],
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 11, 15, 21],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        "Naive Bayes": {
            "model__var_smoothing": [1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
        },
        "SVM": {
            "model__C": [0.1, 1.0, 10.0, 100.0],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto", 0.01, 0.1],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
    }


def tune_group_a_models(X_train, y_train, X_test, y_test):
    """Tune Group A pipelines using CV on train data and evaluate on holdout."""
    results = {}
    best_parameters = {}
    for name, estimator in build_models().items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(X_train)), ("model", estimator)]
        )
        search = RandomizedSearchCV(
            pipeline,
            group_a_search_spaces()[name],
            n_iter=RANDOM_SEARCH_ITERATIONS,
            scoring="f1",
            cv=RANDOM_SEARCH_CV,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        predictions = search.predict(X_test)
        probabilities = search.predict_proba(X_test)[:, 1]
        results[f"{name} tuned"] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": calculate_metrics(y_test, predictions, probabilities),
        }
        best_parameters[name] = search.best_params_
    return results, best_parameters


def advanced_search_spaces():
    """Return random-search spaces for AdaBoost and XGBoost."""
    return {
        "AdaBoost": {
            "n_estimators": [100, 200, 350, 500],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.5],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [2, 3, 4, 5, 7],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "min_child_weight": [1, 2, 5, 10],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "gamma": [0.0, 0.1, 0.5, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "reg_lambda": [1.0, 3.0, 10.0],
        },
    }


def tune_advanced_models(X_train, y_train, X_test, y_test):
    """Random-search Group B models on transformed training data."""
    train_matrix, test_matrix, _ = transform_data(X_train, X_test, X_test)
    candidates = {"AdaBoost": build_adaboost}
    try:
        candidates["XGBoost"] = build_xgboost
    except ImportError as error:
        print(f"Skipping XGBoost random search: {error}")

    results = {}
    best_parameters = {}
    for name, builder in candidates.items():
        try:
            estimator = builder()
        except ImportError as error:
            print(f"Skipping {name} random search: {error}")
            continue
        search = RandomizedSearchCV(
            estimator,
            advanced_search_spaces()[name],
            n_iter=RANDOM_SEARCH_ITERATIONS,
            scoring="f1",
            cv=RANDOM_SEARCH_CV,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(train_matrix, y_train)
        predictions = search.predict(test_matrix)
        probabilities = search.predict_proba(test_matrix)[:, 1]
        results[f"{name} random-search tuned"] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": calculate_metrics(y_test, predictions, probabilities),
        }
        best_parameters[name] = search.best_params_
    return results, best_parameters


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


def write_report(metrics, best_parameters=None, path=REPORT_PATH):
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
    parameter_text = "No tuning results were recorded."
    if best_parameters:
        parameter_lines = [
            f"- **{name}**: `{parameters}`"
            for name, parameters in best_parameters.items()
        ]
        parameter_text = "\n".join(parameter_lines)
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

## Random-search tuning

RandomizedSearchCV selected hyperparameters using {RANDOM_SEARCH_CV}-fold cross-validation
on the training partition. The holdout test partition was used only once for the
final comparison. Search objective: F1 score.

{parameter_text}

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
    tuned_group_a, group_a_parameters = tune_group_a_models(
        X_train, y_train, X_test, y_test
    )
    results.update(tuned_group_a)
    results.update(evaluate_advanced_models(X_train, y_train, X_test, y_test))
    tuned_advanced, advanced_parameters = tune_advanced_models(
        X_train, y_train, X_test, y_test
    )
    results.update(tuned_advanced)
    metrics = save_plots(results, y_test)
    all_parameters = {**group_a_parameters, **advanced_parameters}
    write_report(metrics, all_parameters)
    print(metrics.round(4).to_string())
    print(f"\nPlots written to {OUTPUT_DIR}/")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
