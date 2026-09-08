"""Standalone data cleaning and preparation for patient classification.

The transformer is fitted on known patients only, then applied to both known
and new patients. This keeps imputation, encoding, and scaling leakage-safe.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "cerebrovascular_accident"
TARGET_ALIASES = {"cerebrovascular accident", "cerebrovascular_accident", "stroke"}
ID_COLUMNS = {"id", "patient_id", "patientid"}
RANDOM_STATE = 42


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and fail with a useful path error."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def find_target_column(columns) -> str | None:
    """Find the target using supported spellings, ignoring case and spaces."""
    for column in columns:
        normalized = str(column).strip().lower()
        if normalized in TARGET_ALIASES:
            return column
    return None


def normalize_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename a target alias to the canonical target name when present."""
    output = df.copy()
    target_column = find_target_column(output.columns)
    if target_column and target_column != TARGET_COLUMN:
        output = output.rename(columns={target_column: TARGET_COLUMN})
    return output


def clean_input_data(
    known: pd.DataFrame,
    new: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Normalize target names, remove IDs, and validate known labels."""
    known = normalize_target_column(known)
    new = normalize_target_column(new)
    if TARGET_COLUMN not in known.columns:
        raise ValueError(
            f"Known data must contain one of: {sorted(TARGET_ALIASES)}"
        )

    known = known.dropna(subset=[TARGET_COLUMN]).copy()
    if known[TARGET_COLUMN].nunique() != 2:
        raise ValueError("The target must contain exactly two classes for classification.")

    id_columns = [column for column in known.columns if str(column).lower() in ID_COLUMNS]
    feature_columns = [
        column for column in known.columns
        if column != TARGET_COLUMN and column not in id_columns
    ]
    missing_features = set(feature_columns) - set(new.columns)
    if missing_features:
        raise ValueError(f"New data is missing features: {sorted(missing_features)}")

    return known[feature_columns + [TARGET_COLUMN]], new[feature_columns], feature_columns


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build numeric and nominal-category transformations."""
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("median_imputation", SimpleImputer(strategy="median")),
            ("standard_scaling", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("most_frequent_imputation", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        verbose_feature_names_out=False,
    )


def prepare_data(
    known: pd.DataFrame,
    new: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, ColumnTransformer, dict]:
    """Fit preparation on known features and transform both datasets."""
    known, new, feature_columns = clean_input_data(known, new)
    X_known = known[feature_columns]
    y_known = known[TARGET_COLUMN].astype(int)
    preprocessor = build_preprocessor(X_known)
    X_known_prepared = pd.DataFrame(
        preprocessor.fit_transform(X_known),
        columns=preprocessor.get_feature_names_out(),
        index=X_known.index,
    )
    X_new_prepared = pd.DataFrame(
        preprocessor.transform(new[feature_columns]),
        columns=preprocessor.get_feature_names_out(),
        index=new.index,
    )
    metadata = {
        "target": TARGET_COLUMN,
        "task": "binary classification",
        "input_features": feature_columns,
        "numeric_features": X_known.select_dtypes(include="number").columns.tolist(),
        "categorical_features": X_known.select_dtypes(exclude="number").columns.tolist(),
        "encoded_feature_count": len(X_known_prepared.columns),
        "target_distribution": y_known.value_counts().sort_index().to_dict(),
        "preparation": {
            "numeric_missing_values": "median imputation",
            "categorical_missing_values": "most frequent imputation",
            "categorical_encoder": "one-hot encoding",
            "numeric_scaler": "standard scaling",
            "identifier_columns_removed": sorted(ID_COLUMNS),
        },
    }
    return X_known_prepared, y_known, X_new_prepared, preprocessor, metadata


def save_prepared_outputs(
    X_known: pd.DataFrame,
    y_known: pd.Series,
    X_new: pd.DataFrame,
    metadata: dict,
    output_dir: str | Path = "prepared_data",
) -> None:
    """Save model-ready tables and a preparation metadata report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    known_output = X_known.copy()
    known_output[TARGET_COLUMN] = y_known.to_numpy()
    known_output.to_csv(output_dir / "known_prepared.csv", index=False)
    X_new.to_csv(output_dir / "new_prepared.csv", index=False)
    with (output_dir / "preparation_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)


def main() -> None:
    known = load_csv("Known_Patients_01.csv")
    new = load_csv("New_Patients_01.csv")
    X_known, y_known, X_new, _, metadata = prepare_data(known, new)
    save_prepared_outputs(X_known, y_known, X_new, metadata)
    print("Data preparation complete.")
    print(f"Known prepared shape: {X_known.shape}")
    print(f"New prepared shape: {X_new.shape}")
    print(f"Categorical encoding: {metadata['preparation']['categorical_encoder']}")
    print(f"Numeric scaling: {metadata['preparation']['numeric_scaler']}")
    print("Outputs written to prepared_data/")


if __name__ == "__main__":
    main()
