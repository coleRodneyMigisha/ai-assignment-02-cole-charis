"""Train and compare supervised classifiers for the patient datasets.

The known-patient file contains the binary target ``cerebrovascular_accident``.
The new-patient file is used only after model comparison to generate predictions.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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


def build_models() -> dict[str, object]:
    """Return the six requested classifiers with reproducible settings."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(
            probability=True, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
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
        results.append(
            {
                "model": name,
                "cv_f1_mean": cv_scores["test_f1"].mean(),
                "cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(y_test, predictions, zero_division=0),
                "test_recall": recall_score(y_test, predictions, zero_division=0),
                "test_f1": f1_score(y_test, predictions, zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }
        )
        fitted_pipelines[name] = pipeline

    return pd.DataFrame(results).sort_values("test_f1", ascending=False), fitted_pipelines


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

    print("Task: binary supervised classification")
    print(f"Target: {TARGET_COLUMN} (classes: {sorted(y.unique().tolist())})")
    print("\nModel comparison (ranked by holdout F1):")
    print(results.round(3).to_string(index=False))

    best_model_name = results.iloc[0]["model"]
    best_model = fit_best_model(best_model_name, models, X, y)
    predictions = predict_new_patients(best_model, X_new)
    output = add_predictions(new, predictions)
    output.to_csv("New_Patients_predictions.csv", index=False)
    print(f"\nSelected model: {best_model_name}")
    print("Predictions written to New_Patients_predictions.csv")


if __name__ == "__main__":
    main()