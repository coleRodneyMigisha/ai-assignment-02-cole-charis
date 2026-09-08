"""Run the complete assignment workflow from one command.

Evidently drift analysis is intentionally reported as pending because its
configuration/report policy is being completed separately.
"""

from pathlib import Path
import argparse

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import eda_known_patients
from train_classifiers import (
    RANDOM_STATE,
    build_models,
    build_preprocessor,
    evaluate_models,
    fit_best_model,
    load_datasets,
    prepare_features,
)
from train_advanced_classifiers import (
    build_adaboost,
    build_tuned_adaboost,
    build_tuned_xgboost,
    build_xgboost,
    classification_metrics,
    evaluate_classifiers,
    compare_imbalance_strategies,
    save_metric_visualizations,
    train_lightgbm_focal,
    tune_xgboost,
    transform_data,
)
from xai_analysis import run_xai


OUTPUT_DIR = Path("reports")


def run_eda() -> None:
    """Generate EDA plots with classification task detection."""
    try:
        eda_known_patients.main()
    except Exception as error:
        print(f"EDA did not complete; continuing with the modelling workflow: {error}")


def run_baseline_models(X, y, X_new, new):
    results, _ = evaluate_models(X, y, build_models())
    results.to_csv(OUTPUT_DIR / "baseline_metrics.csv", index=False)
    best_name = results.iloc[0]["model"]
    best_model = fit_best_model(best_name, build_models(), X, y)
    output = new.copy()
    target_name = next(
        (column for column in output if column.strip().lower() == "cerebrovascular accident"),
        "cerebrovascular_accident",
    )
    output[target_name] = best_model.predict(X_new)
    output.to_csv("New_Patients_predictions.csv", index=False)
    return results, best_name, best_model


def run_advanced_models(X, y):
    """Evaluate adjusted Group B models on a held-out test set."""
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )
    train_matrix, test_matrix, _ = transform_data(X_train, X_test, X_test)
    _, validation_matrix, _ = transform_data(X_train, X_validation, X_test)
    models = {
        "AdaBoost": build_adaboost(),
        "AdaBoost adjusted": build_tuned_adaboost(),
        "XGBoost": build_xgboost(),
        "XGBoost adjusted": build_tuned_xgboost(),
    }
    results = evaluate_classifiers(models, train_matrix, y_train, test_matrix, y_test)
    save_metric_visualizations(models, train_matrix, y_train, test_matrix, y_test)
    all_results = results.reset_index().rename(columns={"model": "experiment"})

    try:
        imbalance = compare_imbalance_strategies(train_matrix, y_train, test_matrix, y_test)
        imbalance = imbalance.rename(columns={"strategy": "experiment"})
        all_results = pd.concat([all_results, imbalance], ignore_index=True, sort=False)
    except ImportError as error:
        print(f"Imbalance experiment skipped: {error}")

    try:
        study, tuned = tune_xgboost(
            train_matrix, y_train, validation_matrix, y_validation, n_trials=20
        )
        tuned.fit(train_matrix, y_train)
        tuned_metrics = classification_metrics(
            y_test, tuned.predict_proba(test_matrix)[:, 1]
        )
        all_results = pd.concat(
            [all_results, pd.DataFrame([{**tuned_metrics, "experiment": "XGBoost Optuna tuned"}])],
            ignore_index=True,
            sort=False,
        )
        print(f"Optuna best parameters: {study.best_params}")
    except ImportError as error:
        print(f"Optuna experiment skipped: {error}")

    try:
        _, focal_metrics = train_lightgbm_focal(
            train_matrix, y_train, test_matrix, y_test
        )
        all_results = pd.concat(
            [all_results, pd.DataFrame([{**focal_metrics, "experiment": "LightGBM focal loss"}])],
            ignore_index=True,
            sort=False,
        )
    except ImportError as error:
        print(f"LightGBM experiment skipped: {error}")

    all_results.to_csv(OUTPUT_DIR / "advanced_metrics.csv", index=False)
    best_name = results.index[0]
    return results, best_name


def build_group_b_pipeline(model_name, X, y):
    """Fit the selected Group B estimator behind the raw-data preprocessor."""
    builders = {
        "AdaBoost": build_adaboost,
        "AdaBoost adjusted": build_tuned_adaboost,
        "XGBoost": build_xgboost,
        "XGBoost adjusted": build_tuned_xgboost,
    }
    if model_name not in builders:
        raise ValueError(f"Unsupported Group B model: {model_name}")
    pipeline = Pipeline(
        [("preprocessor", build_preprocessor(X)), ("model", builders[model_name]())]
    )
    return pipeline.fit(X, y)


def main(run_eda_stage: bool = True) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    known, new = load_datasets()
    X, y, X_new = prepare_features(known, new)

    print("=== 1. EDA ===")
    if run_eda_stage:
        run_eda()
    else:
        print("EDA skipped; existing plots are preserved.")

    print("\n=== 2. Baseline classifiers: Group A ===")
    baseline_results, baseline_name, baseline_model = run_baseline_models(X, y, X_new, new)
    print(baseline_results.round(3).to_string(index=False))
    print(f"Best Group A model: {baseline_name}")

    print("\n=== 3. Advanced classifiers: Group B ===")
    advanced_results, advanced_name = run_advanced_models(X, y)
    print(advanced_results.round(3).to_string())
    print(f"Best Group B model: {advanced_name}")

    print("\n=== 4. XAI ===")
    xai_results = run_xai(baseline_model, X, model_name="group_a_best")
    print("Group A XAI status:", list(xai_results))
    group_b_model = build_group_b_pipeline(advanced_name, X, y)
    group_b_xai = run_xai(group_b_model, X, model_name="group_b_best")
    print("Group B XAI status:", list(group_b_xai))

    print("\n=== 5. Evidently ===")
    print("Pending: Evidently drift configuration is intentionally left for the project owner.")
    print("\nReports written to reports/ and plots/. Predictions written to New_Patients_predictions.csv.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the patient classification assignment.")
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip the slower EDA plotting stage and run modelling, XAI, and reports.",
    )
    arguments = parser.parse_args()
    main(run_eda_stage=not arguments.skip_eda)
