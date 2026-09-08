"""Model explainability utilities for the two classifier groups.

SHAP and LIME are optional. Counterfactual search is implemented locally for
this tabular binary problem and reports the smallest one-feature changes that
flip a model prediction when a flip can be found.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def _transformed_model(model, X):
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = preprocessor.get_feature_names_out()
    return estimator, transformed, names


def generate_shap_explanation(model, X, output_dir="plots/xai", model_name="model"):
    """Save a SHAP global feature-importance plot when SHAP is installed."""
    try:
        import shap
    except ImportError as error:
        raise ImportError("Install SHAP to generate explanations: pip install shap") from error

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    estimator, transformed, feature_names = _transformed_model(model, X)
    background = transformed[: min(100, len(transformed))]
    sample = transformed[: min(300, len(transformed))]
    if hasattr(estimator, "predict_proba"):
        predict = estimator.predict_proba
    else:
        predict = estimator.predict
    try:
        explainer = shap.Explainer(predict, background, feature_names=feature_names)
        values = explainer(sample)
        shap_values = values.values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        importance = np.abs(shap_values).mean(axis=0)
        result = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        result.sort_values("mean_abs_shap", ascending=False).to_csv(
            output_path / f"{model_name}_shap_importance.csv", index=False
        )
        shap.summary_plot(values, sample, feature_names=feature_names, show=False, plot_type="bar")
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig(output_path / f"{model_name}_shap_summary.png", dpi=150)
        plt.close()
        return result
    except Exception as error:
        raise RuntimeError(f"SHAP explanation failed for {model_name}: {error}") from error


def generate_lime_explanation(model, X, row_index=0, output_dir="plots/xai", model_name="model"):
    """Save a local LIME explanation for one patient when LIME is installed."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as error:
        raise ImportError("Install LIME to generate explanations: pip install lime") from error

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    estimator, transformed, feature_names = _transformed_model(model, X)
    explainer = LimeTabularExplainer(
        transformed,
        feature_names=feature_names,
        class_names=["no_event", "event"],
        mode="classification",
        random_state=42,
    )
    explanation = explainer.explain_instance(
        transformed[row_index], estimator.predict_proba, num_features=15
    )
    explanation.save_to_file(str(output_path / f"{model_name}_lime.html"))
    return explanation.as_list()


def generate_counterfactuals(model, X, output_dir="plots/xai", model_name="model"):
    """Find simple one-feature counterfactuals for the first positive patient."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(X)[:, 1]
    source_index = int(np.argmax(probabilities))
    source = X.iloc[source_index].copy()
    source_prediction = int(probabilities[source_index] >= 0.5)
    candidates = []
    numeric_columns = X.select_dtypes(include="number").columns
    for column in numeric_columns:
        values = X[column].dropna()
        for value in np.linspace(values.quantile(0.1), values.quantile(0.9), 9):
            candidate = source.copy()
            candidate[column] = value
            candidate_probability = float(model.predict_proba(pd.DataFrame([candidate]))[0, 1])
            candidate_prediction = int(candidate_probability >= 0.5)
            if candidate_prediction != source_prediction:
                distance = abs(float(source[column]) - float(value)) / (values.std() or 1.0)
                candidates.append(
                    {
                        "source_index": source_index,
                        "feature": column,
                        "original_value": source[column],
                        "counterfactual_value": value,
                        "original_probability": probabilities[source_index],
                        "counterfactual_probability": candidate_probability,
                        "normalized_change": distance,
                    }
                )
    result = pd.DataFrame(candidates).sort_values("normalized_change")
    result.to_csv(output_path / f"{model_name}_counterfactuals.csv", index=False)
    return result


def run_xai(model, X, output_dir="plots/xai", model_name="model"):
    """Run available XAI methods without making optional packages mandatory."""
    results = {}
    try:
        results["shap"] = generate_shap_explanation(model, X, output_dir, model_name)
    except (ImportError, RuntimeError) as error:
        results["shap_error"] = str(error)
    try:
        results["lime"] = generate_lime_explanation(model, X, 0, output_dir, model_name)
    except ImportError as error:
        results["lime_error"] = str(error)
    results["counterfactuals"] = generate_counterfactuals(model, X, output_dir, model_name)
    return results
