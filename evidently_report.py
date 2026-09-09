"""Generate Evidently monitoring reports for the patient classifier.

Run from this directory:

    python evidently_report.py

Or specify input/output locations:

    python evidently_report.py --known Known_Patients_01.csv \
        --new New_Patients_01.csv --output reports/evidently

The new-patient file does not contain observed outcomes. Therefore the script
reports feature drift and prediction drift for the new cohort, while the
classification-performance report uses a labelled train/holdout comparison.
True current-cohort model quality requires outcomes collected later.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import pandas as pd

# Avoid the Windows WMIC warning emitted by joblib during model comparison.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# Make the script runnable from the project root as well as from this directory.
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evidently_monitoring import run_evidently_monitoring
from train_classifiers import (
    build_models,
    evaluate_models,
    fit_best_model,
    prepare_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the selected classifier and generate Evidently reports."
    )
    parser.add_argument(
        "--known",
        type=Path,
        default=PROJECT_DIR / "Known_Patients_01.csv",
        help="Labelled reference dataset CSV.",
    )
    parser.add_argument(
        "--new",
        type=Path,
        default=PROJECT_DIR / "New_Patients_01.csv",
        help="Current/new-patient dataset CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "reports" / "evidently",
        help="Directory in which HTML and JSON reports are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.known.exists():
        raise FileNotFoundError(f"Known dataset not found: {args.known}")
    if not args.new.exists():
        raise FileNotFoundError(f"New dataset not found: {args.new}")

    known = pd.read_csv(args.known)
    new = pd.read_csv(args.new)
    X, y, X_new = prepare_features(known, new)

    print("Comparing candidate classifiers...")
    results, _ = evaluate_models(X, y, build_models())
    model_name = str(results.iloc[0]["model"])
    model = fit_best_model(model_name, build_models(), X, y)
    print(f"Selected model: {model_name}")

    summary = run_evidently_monitoring(
        known=known,
        new=new,
        model=model,
        X=X,
        y=y,
        X_new=X_new,
        output_dir=args.output,
        model_name=model_name,
    )

    print(f"Reports written to: {args.output.resolve()}")
    for report_name in summary["reports"].values():
        print(f"  - {report_name}")
    print(
        "Note: new-patient labels are unavailable; prediction drift is a proxy "
        "until ground-truth outcomes are collected."
    )


if __name__ == "__main__":
    main()
