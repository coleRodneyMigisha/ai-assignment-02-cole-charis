"""Advanced imbalance experiments for the patient classification task.

This module compares AdaBoost and XGBoost, measures XGBoost with and without
SMOTE, optionally tunes XGBoost with Optuna, and trains LightGBM with a custom
binary focal-loss objective. Install optional dependencies with:

    pip install xgboost optuna imbalanced-learn lightgbm
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from train_classifiers import (
    RANDOM_STATE,
    TARGET_COLUMN,
    build_preprocessor,
    load_datasets,
    prepare_features,
)


def transform_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_new: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit preprocessing on training data and return dense numeric matrices."""
    preprocessor = build_preprocessor(X_train)
    train_matrix = preprocessor.fit_transform(X_train)
    test_matrix = preprocessor.transform(X_test)
    new_matrix = preprocessor.transform(X_new)
    return (
        train_matrix.toarray() if hasattr(train_matrix, "toarray") else train_matrix,
        test_matrix.toarray() if hasattr(test_matrix, "toarray") else test_matrix,
        new_matrix.toarray() if hasattr(new_matrix, "toarray") else new_matrix,
    )


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate metrics that expose the precision/recall trade-off."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
    }


def build_adaboost() -> AdaBoostClassifier:
    """Build a reproducible AdaBoost classifier."""
    return AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE)


def require_xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError("Install xgboost to run the advanced classifiers.") from error
    return XGBClassifier


def build_xgboost(**overrides):
    """Build XGBoost with imbalance-aware defaults and tunable parameters."""
    XGBClassifier = require_xgboost()
    settings = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    settings.update(overrides)
    return XGBClassifier(**settings)


def compare_imbalance_strategies(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compare class weighting, SMOTE, and a lower decision threshold."""
    strategies = {}
    weighted = build_xgboost(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
    weighted.fit(X_train, y_train)
    strategies["XGBoost weighted"] = weighted.predict_proba(X_test)[:, 1]

    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as error:
        raise ImportError(
            "Install imbalanced-learn to compare SMOTE: pip install imbalanced-learn"
        ) from error

    smote_X, smote_y = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    smote_model = build_xgboost()
    smote_model.fit(smote_X, smote_y)
    strategies["XGBoost + SMOTE"] = smote_model.predict_proba(X_test)[:, 1]

    rows = []
    for name, probabilities in strategies.items():
        for threshold in (0.5, 0.35):
            row = classification_metrics(y_test, probabilities, threshold)
            row.update({"strategy": name, "threshold": threshold})
            rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def tune_xgboost(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_validation: np.ndarray,
    y_validation: pd.Series,
    n_trials: int = 20,
):
    """Tune tree, learning-rate, sampling, and regularization parameters."""
    try:
        import optuna
    except ImportError as error:
        raise ImportError("Install Optuna to tune XGBoost: pip install optuna") from error

    positive_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        model = build_xgboost(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            gamma=trial.suggest_float("gamma", 0, 5),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            scale_pos_weight=positive_weight,
        )
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_validation)[:, 1]
        return roc_auc_score(y_validation, probabilities)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study, build_xgboost(**study.best_params, scale_pos_weight=positive_weight)


def focal_loss_objective(alpha: float = 0.75, gamma: float = 2.0):
    """Return a LightGBM focal-loss objective with numerical Hessian."""
    def objective(predictions, dataset):
        labels = dataset.get_label()
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(predictions, -35, 35)))
        positive = labels == 1
        weights = np.where(positive, alpha, 1.0 - alpha)
        gradients = np.where(
            positive,
            weights * (gamma * probabilities * (1 - probabilities) ** gamma * np.log(np.maximum(probabilities, 1e-12)) - (1 - probabilities) ** (gamma + 1)),
            weights * (gamma * probabilities**gamma * (1 - probabilities) * -np.log(np.maximum(1 - probabilities, 1e-12)) + probabilities ** (gamma + 1)),
        )
        step = 1e-3
        shifted = probabilities + step * probabilities * (1 - probabilities)
        shifted = np.clip(shifted, 1e-7, 1 - 1e-7)
        shifted_gradient = np.where(
            positive,
            weights * (gamma * shifted * (1 - shifted) ** gamma * np.log(shifted) - (1 - shifted) ** (gamma + 1)),
            weights * (gamma * shifted**gamma * (1 - shifted) * -np.log(1 - shifted) + shifted ** (gamma + 1)),
        )
        hessians = np.maximum(np.abs((shifted_gradient - gradients) / step), 1e-6)
        return gradients, hessians

    return objective


def train_lightgbm_focal(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    alpha: float = 0.75,
    gamma: float = 2.0,
):
    """Train LightGBM with focal loss and return model plus evaluation metrics."""
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise ImportError("Install LightGBM to run focal loss: pip install lightgbm") from error

    train_set = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(
        {
            "objective": focal_loss_objective(alpha, gamma),
            "learning_rate": 0.04,
            "num_leaves": 31,
            "verbosity": -1,
            "seed": RANDOM_STATE,
        },
        train_set,
        num_boost_round=200,
    )
    raw_predictions = model.predict(X_test)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw_predictions, -35, 35)))
    return model, classification_metrics(y_test, probabilities)


def main(n_trials: int = 20) -> None:
    known, new = load_datasets()
    X, y, X_new = prepare_features(known, new)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    train_matrix, test_matrix, new_matrix = transform_data(X_train, X_test, X_new)

    adaboost = build_adaboost().fit(train_matrix, y_train)
    ada_metrics = classification_metrics(y_test, adaboost.predict_proba(test_matrix)[:, 1])
    print("AdaBoost:", {key: round(value, 3) for key, value in ada_metrics.items()})

    print("\nImbalance comparison:")
    print(compare_imbalance_strategies(train_matrix, y_train, test_matrix, y_test).round(3).to_string(index=False))

    study, tuned_model = tune_xgboost(train_matrix, y_train, test_matrix, y_test, n_trials)
    tuned_model.fit(train_matrix, y_train)
    tuned_metrics = classification_metrics(y_test, tuned_model.predict_proba(test_matrix)[:, 1])
    print("\nBest XGBoost parameters:", study.best_params)
    print("Tuned XGBoost:", {key: round(value, 3) for key, value in tuned_metrics.items()})

    _, focal_metrics = train_lightgbm_focal(train_matrix, y_train, test_matrix, y_test)
    print("LightGBM focal loss:", {key: round(value, 3) for key, value in focal_metrics.items()})

    full_matrix, _, new_matrix = transform_data(X, X, X_new)
    tuned_model.fit(full_matrix, y)
    new_predictions = tuned_model.predict(new_matrix)
    output = new.copy()
    output_target = next((column for column in output if column.strip().lower() == TARGET_COLUMN.replace("_", " ")), TARGET_COLUMN)
    output[output_target] = new_predictions
    output.to_csv("New_Patients_advanced_predictions.csv", index=False)
    print("\nPredictions written to New_Patients_advanced_predictions.csv")


if __name__ == "__main__":
    main()