"""Evidently monitoring for the cerebrovascular-accident classifier.

The new-patient file has no ground-truth labels, so this module separates:

* feature/data drift: known patients (reference) versus new patients (current);
* prediction drift: model predictions for known versus new patients; and
* model-performance drift/generalisation: labelled train versus holdout data.

The first two are available before labels arrive. True production model drift
requires labels for the current cohort; the generated report states this
limitation explicitly instead of presenting prediction drift as accuracy drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from train_classifiers import RANDOM_STATE

TARGET_COLUMN = "cerebrovascular_accident"
PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN = "prediction_proba"
ID_COLUMNS = {"id", "patient_id", "patientid"}


def _evidently_report(presets: list[Any], current: pd.DataFrame, reference: pd.DataFrame | None):
    """Run a report using the current Evidently API."""
    from evidently import Report

    report = Report(metrics=presets)
    try:
        return report.run(current=current, reference=reference)
    except TypeError as error:
        if "unexpected keyword argument 'current'" not in str(error):
            raise
        return report.run(current, reference)


def _save_report(evaluation: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.save_html(str(path))


def _normalise_target(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    aliases = {"cerebrovascular accident", "cerebrovascular_accident", "stroke"}
    target = next((c for c in output.columns if str(c).strip().lower() in aliases), None)
    if target and target != TARGET_COLUMN:
        output = output.rename(columns={target: TARGET_COLUMN})
    return output


def _monitoring_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return raw input features, excluding IDs and target labels."""
    output = _normalise_target(df)
    excluded = ID_COLUMNS | {TARGET_COLUMN}
    return output[[c for c in output.columns if str(c).lower() not in excluded]].copy()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _extract_summary(evaluation: Any) -> dict[str, Any]:
    """Extract Evidently's top-level summary without depending on metric IDs."""
    try:
        snapshot = evaluation.dict()
    except AttributeError:
        try:
            snapshot = evaluation.json()
        except AttributeError:
            return {}
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except json.JSONDecodeError:
            return {"raw": snapshot}
    return _json_safe(snapshot)


def run_evidently_monitoring(
    known: pd.DataFrame,
    new: pd.DataFrame,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    X_new: pd.DataFrame,
    output_dir: str | Path = "reports/evidently",
    model_name: str = "selected_model",
) -> dict[str, Any]:
    """Generate Evidently reports and return a concise run summary.

    Parameters are the raw known/new frames plus the already fitted selected
    model and aligned feature frames used by the existing training pipeline.
    """
    try:
        from evidently.presets import Classification, DataDrift
    except ImportError as error:
        try:
            from evidently.presets import ClassificationPreset as Classification
            from evidently.presets import DataDriftPreset as DataDrift
        except ImportError:
            raise RuntimeError(
                "Evidently is required. Install project dependencies with "
                "`pip install -r requirements.txt`."
            ) from error

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature/data drift: raw, interpretable clinical variables.
    reference_features = _monitoring_features(known)
    current_features = _monitoring_features(new)
    data_eval = _evidently_report([DataDrift()], current_features, reference_features)
    _save_report(data_eval, output_dir / "data_drift.html")

    # 2. Prediction drift: available even though new labels are unknown.
    known_predictions = np.asarray(model.predict(X)).astype(int)
    new_predictions = np.asarray(model.predict(X_new)).astype(int)
    known_probabilities = np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    new_probabilities = np.asarray(model.predict_proba(X_new)[:, 1], dtype=float)
    reference_prediction_data = reference_features.copy()
    current_prediction_data = current_features.copy()
    reference_prediction_data[PREDICTION_COLUMN] = known_predictions
    current_prediction_data[PREDICTION_COLUMN] = new_predictions
    reference_prediction_data[PROBABILITY_COLUMN] = known_probabilities
    current_prediction_data[PROBABILITY_COLUMN] = new_probabilities
    prediction_eval = _evidently_report(
        [DataDrift()], current_prediction_data, reference_prediction_data
    )
    _save_report(prediction_eval, output_dir / "prediction_drift.html")

    # 3. Labelled performance/generalisation drift: train versus holdout.
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    monitoring_model = clone(model).fit(X_train, y_train)
    train_pred = np.asarray(monitoring_model.predict(X_train)).astype(int)
    holdout_pred = np.asarray(monitoring_model.predict(X_holdout)).astype(int)
    train_proba = np.asarray(monitoring_model.predict_proba(X_train)[:, 1], dtype=float)
    holdout_proba = np.asarray(monitoring_model.predict_proba(X_holdout)[:, 1], dtype=float)
    performance_reference = pd.DataFrame(
        {"target": np.asarray(y_train).astype(int), PREDICTION_COLUMN: train_pred, PROBABILITY_COLUMN: train_proba}
    )
    performance_current = pd.DataFrame(
        {"target": np.asarray(y_holdout).astype(int), PREDICTION_COLUMN: holdout_pred, PROBABILITY_COLUMN: holdout_proba}
    )
    from evidently import BinaryClassification, DataDefinition, Dataset

    classification_definition = DataDefinition(
        classification=[
            BinaryClassification(
                target="target",
                prediction_labels=PREDICTION_COLUMN,
                prediction_probas=PROBABILITY_COLUMN,
                pos_label=1,
            )
        ]
    )
    performance_reference_dataset = Dataset.from_pandas(
        performance_reference, data_definition=classification_definition
    )
    performance_current_dataset = Dataset.from_pandas(
        performance_current, data_definition=classification_definition
    )
    performance_eval = _evidently_report(
        [Classification()], performance_current_dataset, performance_reference_dataset
    )
    _save_report(performance_eval, output_dir / "model_performance_drift.html")

    summary = {
        "model": model_name,
        "reference_dataset": "Known_Patients_01.csv",
        "current_dataset": "New_Patients_01.csv",
        "reference_rows": int(len(known)),
        "current_rows": int(len(new)),
        "known_positive_prediction_rate": float(known_predictions.mean()),
        "new_positive_prediction_rate": float(new_predictions.mean()),
        "known_mean_predicted_probability": float(known_probabilities.mean()),
        "new_mean_predicted_probability": float(new_probabilities.mean()),
        "performance_evaluation": {
            "reference": "labelled training partition",
            "current": "labelled stratified holdout partition",
            "note": "This is a generalisation/performance comparison. New-patient model quality cannot be measured until ground-truth outcomes are collected.",
        },
        "reports": {
            "data_drift": "data_drift.html",
            "prediction_drift": "prediction_drift.html",
            "model_performance_drift": "model_performance_drift.html",
        },
        "evidently": {
            "data_drift_summary": _extract_summary(data_eval),
            "prediction_drift_summary": _extract_summary(prediction_eval),
            "model_performance_summary": _extract_summary(performance_eval),
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(_json_safe(summary), file, indent=2)

    return summary


if __name__ == "__main__":
    raise SystemExit("Run Evidently through main.py so it uses the selected trained model.")
