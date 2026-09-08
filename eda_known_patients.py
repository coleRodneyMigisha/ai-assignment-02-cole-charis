"""
EDA and preprocessing script for Known_Patients_01 (cerebrovascular accident data).

Generates >5 plots (many more, in practice), inspects bias/variance/noise,
correlation between variables (numeric-numeric, numeric-target, categorical-target),
recommends encoders, and checks linearity.

Run: python eda_known_patients.py [path_to_file.csv_or_xlsx]
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency, pointbiserialr, ks_2samp

from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, log_loss

sns.set(style='whitegrid')

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

DEFAULT_PATH = 'Known_Patients_01.xlsx'


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def infer_target(df):
    candidates = ['target', 'label', 'outcome', 'diagnosis', 'result',
                   'cerebrovascular accident', 'stroke']
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[-1]


def load_data(path=None):
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        print(f"Error: {path} not found in working directory.")
        sys.exit(1)
    if path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # drop obvious ID columns - they are noise for modeling, not signal
    id_like = [c for c in df.columns if c.lower() in ('id', 'patient_id', 'patientid')]
    if id_like:
        print(f'Dropping ID-like columns from analysis: {id_like}')
        df = df.drop(columns=id_like)
    return df


def quick_overview(df):
    print('\n=== Data Overview ===')
    print('Rows, cols:', df.shape)
    print('\nHead:')
    print(df.head())
    print('\nDtypes:')
    print(df.dtypes)
    print('\nMissing values per column:')
    print(df.isnull().sum())
    print('\nDescribe numeric:')
    print(df.describe().T)


def infer_column_types(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    # treat low-cardinality numerics (flags like 0/1) as categorical
    for col in numeric_cols[:]:
        if df[col].nunique() <= 10:
            obj_cols.append(col)
            numeric_cols.remove(col)
    return numeric_cols, obj_cols


# ---------------------------------------------------------------------------
# EDA plots (baseline distributions)
# ---------------------------------------------------------------------------

def plot_missingness(df):
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print('No missing values to plot.')
        return
    plt.figure(figsize=(6, 4))
    miss.sort_values().plot(kind='barh')
    plt.title('Missing values per column')
    plt.xlabel('Count missing')
    plt.savefig(os.path.join(PLOTS_DIR, 'missingness.png'), bbox_inches='tight')
    plt.close()


def plot_distributions(df, numeric_cols):
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(PLOTS_DIR, f'dist_{col}.png'), bbox_inches='tight')
        plt.close()


def plot_boxplots(df, numeric_cols):
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col].dropna())
        plt.title(f'Boxplot of {col}')
        plt.savefig(os.path.join(PLOTS_DIR, f'box_{col}.png'), bbox_inches='tight')
        plt.close()


def plot_categorical_counts(df, cat_cols):
    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Counts of {col}')
        plt.savefig(os.path.join(PLOTS_DIR, f'count_{col}.png'), bbox_inches='tight')
        plt.close()


def plot_pairplot(df, cols, hue=None):
    sub = df[cols].dropna() if hue is None else df[cols + [hue]].dropna()
    sns.pairplot(sub, corner=True, hue=hue)
    plt.savefig(os.path.join(PLOTS_DIR, 'pairplot.png'), bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Correlation between variables
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation heatmap (numeric-numeric)')
    plt.savefig(os.path.join(PLOTS_DIR, 'corr_heatmap.png'), bbox_inches='tight')
    plt.close()
    return corr


def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return np.sqrt(phi2corr / denom) if denom > 0 else np.nan


def categorical_correlation_matrix(df, cat_cols):
    """Cramer's V between every pair of categorical columns -> captures
    association the numeric correlation heatmap can't see."""
    n = len(cat_cols)
    mat = pd.DataFrame(np.eye(n), index=cat_cols, columns=cat_cols)
    for i, c1 in enumerate(cat_cols):
        for j, c2 in enumerate(cat_cols):
            if j <= i:
                continue
            ct = pd.crosstab(df[c1], df[c2])
            v = cramers_v(ct.values)
            mat.loc[c1, c2] = v
            mat.loc[c2, c1] = v
    plt.figure(figsize=(7, 6))
    sns.heatmap(mat.astype(float), annot=True, fmt='.2f', cmap='viridis')
    plt.title("Categorical-categorical association (Cramer's V)")
    plt.savefig(os.path.join(PLOTS_DIR, 'corr_categorical.png'), bbox_inches='tight')
    plt.close()
    return mat


def indicator_feature_analysis(df, numeric_cols, cat_cols, target_col, task):
    """For each candidate predictor, quantify how strongly it 'indicates' the
    target: point-biserial correlation / t-test for numeric vs binary target,
    Cramer's V / chi-square for categorical vs target. Produces a ranked bar
    chart, which is the fastest way to see which features actually carry
    signal about the outcome."""
    print('\n=== Indicator feature strength vs target ===')
    scores = {}

    is_binary_target = df[target_col].nunique(dropna=True) == 2

    for col in numeric_cols:
        sub = df[[col, target_col]].dropna()
        if sub.shape[0] < 10:
            continue
        if is_binary_target:
            r, p = pointbiserialr(sub[target_col], sub[col])
            scores[col] = (abs(r), p, 'point-biserial r')
            # split boxplot: does this feature look different across target classes?
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=target_col, y=col, data=sub)
            plt.title(f'{col} by {target_col} (r={r:.2f}, p={p:.4f})')
            plt.savefig(os.path.join(PLOTS_DIR, f'target_box_{col}.png'), bbox_inches='tight')
            plt.close()
        else:
            r, p = stats.pearsonr(sub[col], sub[target_col])
            scores[col] = (abs(r), p, 'pearson r')

    for col in cat_cols:
        if col == target_col:
            continue
        sub = df[[col, target_col]].dropna()
        ct = pd.crosstab(sub[col], sub[target_col])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(ct)
        v = cramers_v(ct.values)
        scores[col] = (v, p, "Cramer's V")
        if is_binary_target:
            rate = sub.groupby(col)[target_col].mean().sort_values(ascending=False)
            plt.figure(figsize=(6, 4))
            rate.plot(kind='bar', color='indianred')
            plt.title(f'Positive rate of {target_col} by {col} (p={p:.4f})')
            plt.ylabel(f'mean {target_col}')
            plt.savefig(os.path.join(PLOTS_DIR, f'target_rate_{col}.png'), bbox_inches='tight')
            plt.close()

    ranked = sorted(scores.items(), key=lambda x: -x[1][0])
    print(f"{'feature':25s} {'assoc.':>8s} {'p-value':>10s}  metric")
    for k, (v, p, metric) in ranked:
        print(f'{k:25s} {v:8.3f} {p:10.4g}  {metric}')

    plt.figure(figsize=(7, max(4, 0.4 * len(ranked))))
    names = [k for k, _ in ranked]
    vals = [v for _, (v, p, m) in scores.items() for k in [None]][:0]  # noop to keep lints quiet
    vals = [scores[k][0] for k in names]
    sns.barplot(x=vals, y=names, orient='h')
    plt.title(f'Indicator feature strength vs {target_col}')
    plt.xlabel('Association strength (|r| or Cramer V)')
    plt.savefig(os.path.join(PLOTS_DIR, 'indicator_feature_strength.png'), bbox_inches='tight')
    plt.close()

    return ranked


# ---------------------------------------------------------------------------
# Encoders / linearity
# ---------------------------------------------------------------------------

def recommend_encoders(df, cat_cols):
    recommendations = {}
    for col in cat_cols:
        nunique = df[col].nunique(dropna=False)
        if nunique <= 10:
            recommendations[col] = 'OneHotEncoder (nominal, low cardinality)'
        else:
            recommendations[col] = 'Target encoding / frequency encoding (high cardinality)'
    print('\nEncoder recommendations:')
    for k, v in recommendations.items():
        print(f'- {k}: {v}')
    return recommendations


def scatter_with_reg(df, x, y):
    plt.figure(figsize=(6, 4))
    sns.regplot(x=x, y=y, data=df, scatter_kws={'s': 10}, line_kws={'color': 'red'})
    plt.title(f'{y} vs {x} (with linear fit)')
    plt.savefig(os.path.join(PLOTS_DIR, f'scatter_{y}_vs_{x}.png'), bbox_inches='tight')
    plt.close()


def simple_linearity_check(df, numeric_cols, target_col):
    results = {}
    for col in numeric_cols:
        if col == target_col:
            continue
        sub = df[[col, target_col]].dropna()
        if sub.shape[0] < 10:
            continue
        x = sub[[col]].values
        y = sub[target_col].values
        lr = LinearRegression()
        try:
            lr.fit(x, y)
            ypred = lr.predict(x)
            r2 = r2_score(y, ypred)
            results[col] = r2
            scatter_with_reg(sub, col, target_col)
        except Exception:
            continue
    print('\nLinearity (R^2 of a single-feature linear fit) by numeric predictor:')
    for k, v in sorted(results.items(), key=lambda x: -x[1])[:10]:
        print(f'- {k}: R^2={v:.3f}')
    print('Low R^2 here does not rule out a feature - it only means the *raw* '
          'relationship with the target is not linear. Tree-based models can '
          'still use the feature via splits/interactions.')
    return results


# ---------------------------------------------------------------------------
# Bias / Variance / Noise investigation
# ---------------------------------------------------------------------------

def analyze_bias_variance_noise(X, y, task='classification'):
    """
    Bias: systematic underfitting -> a simple model (LogisticRegression /
          LinearRegression) scoring poorly on BOTH train and test.
    Variance: overfitting -> a flexible model (RandomForest) scoring near-
          perfect on train but much worse on test, and/or a wide spread of
          cross-validation fold scores.
    Noise (irreducible error): estimated as the gap that remains even for the
          most flexible model once bias and variance are accounted for -
          approximated here via (a) cross-val score dispersion, (b) label
          consistency for near-duplicate feature rows, and (c) the ceiling
          score of a very flexible model.
    """
    print('\n=== Bias / Variance / Noise Investigation ===')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task == 'classification' else None
    )

    if task == 'regression':
        simple_model = LinearRegression()
        flexible_model = RandomForestRegressor(n_estimators=200, random_state=42)
        score_fn = r2_score
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'r2'
    else:
        simple_model = LogisticRegression(max_iter=2000)
        flexible_model = RandomForestClassifier(n_estimators=200, random_state=42)
        score_fn = accuracy_score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'accuracy'

    if task == 'classification':
        pos_rate = y.mean() if y.nunique() == 2 else None
        if pos_rate is not None and (pos_rate < 0.15 or pos_rate > 0.85):
            print(f'NOTE: target is imbalanced (positive rate={pos_rate:.3f}). '
                  f'A model that always predicts the majority class scores '
                  f'accuracy={max(pos_rate, 1 - pos_rate):.3f} without learning anything. '
                  f'Accuracy below is reported for consistency with the "bias/variance via '
                  f'train-vs-test score" framing, but ROC-AUC (also printed) is the metric '
                  f'to trust here.')

    results = {}
    for name, m in [('Simple (bias-prone)', simple_model), ('Flexible (variance-prone)', flexible_model)]:
        m.fit(X_train, y_train)
        train_score = score_fn(y_train, m.predict(X_train))
        test_score = score_fn(y_test, m.predict(X_test))
        cv_scores = cross_val_score(m, X, y, cv=cv, scoring=scoring)

        gap = train_score - test_score
        print(f'\n{name} [{m.__class__.__name__}]')
        print(f'  train_score={train_score:.3f}  test_score={test_score:.3f}  '
              f'gap={gap:.3f}')
        print(f'  cross-val scores: {np.round(cv_scores, 3)}  '
              f'mean={cv_scores.mean():.3f}  std={cv_scores.std():.3f}')

        if task == 'classification' and hasattr(m, 'predict_proba'):
            train_auc = roc_auc_score(y_train, m.predict_proba(X_train)[:, 1])
            test_auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
            print(f'  ROC-AUC: train={train_auc:.3f}  test={test_auc:.3f}  '
                  f'(0.5=random, 1.0=perfect - use this over accuracy on imbalanced data)')
        if gap > 0.1:
            print('  -> Large train/test gap suggests HIGH VARIANCE (overfitting).')
        if train_score < 0.7 and test_score < 0.7:
            print('  -> Both scores low suggests HIGH BIAS (underfitting) for this model class.')
        if cv_scores.std() > 0.05:
            print('  -> High spread across CV folds is itself evidence of variance '
                  '(the model/data combo is sensitive to which rows land in train vs test).')

        results[name] = dict(train_score=train_score, test_score=test_score,
                              gap=gap, cv_scores=cv_scores)

        # learning curve: train vs cv score as sample size grows
        plt.figure(figsize=(6, 4))
        train_sizes, train_scores, test_scores = learning_curve(
            m, X, y, cv=cv, scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 5)
        )
        plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='train')
        plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='cv')
        plt.xlabel('Training examples')
        plt.ylabel(scoring)
        plt.title(f'Learning curve: {m.__class__.__name__}')
        plt.legend()
        plt.savefig(os.path.join(PLOTS_DIR, f'learning_curve_{m.__class__.__name__}.png'),
                     bbox_inches='tight')
        plt.close()

        # spread of CV scores -> visual proxy for variance
        plt.figure(figsize=(4, 4))
        sns.boxplot(y=cv_scores)
        plt.title(f'CV score spread: {m.__class__.__name__}\n(wide box = high variance)')
        plt.savefig(os.path.join(PLOTS_DIR, f'cv_spread_{m.__class__.__name__}.png'),
                     bbox_inches='tight')
        plt.close()

    # --- Noise estimate ---
    print('\n--- Noise / irreducible error estimate ---')
    flex_test_score = results['Flexible (variance-prone)']['test_score']
    if task == 'classification':
        ceiling = 1.0
    else:
        ceiling = 1.0
    residual_gap = ceiling - flex_test_score
    print(f'Even the flexible model tops out at test_score={flex_test_score:.3f} '
          f'(vs a theoretical ceiling of {ceiling:.1f}).')
    print(f'That remaining gap (~{residual_gap:.3f}) is the combination of noise in the '
          'labels/features plus any signal not captured by these predictors - it is '
          'a rough upper bound on the irreducible error, since RandomForest can fit '
          'almost arbitrary structure.')

    return results


def duplicate_conflict_check(df, feature_cols, target_col):
    """Rows with identical feature values but different target labels are
    direct, unambiguous evidence of label/measurement noise - the model
    literally cannot get these right for both rows at once."""
    print('\n=== Duplicate-feature / conflicting-label check (direct noise evidence) ===')
    sub = df[feature_cols + [target_col]].copy()
    sub_features_only = df[feature_cols]
    dup_mask = sub_features_only.duplicated(keep=False)
    n_dup_rows = dup_mask.sum()
    if n_dup_rows == 0:
        print('No rows share identical feature values - no direct duplicate-label evidence found.')
        return None
    dup_groups = sub[dup_mask].groupby(feature_cols)[target_col].nunique()
    conflicting = dup_groups[dup_groups > 1]
    print(f'{n_dup_rows} rows share feature values with at least one other row.')
    print(f'{len(conflicting)} of those feature-value groups have MORE THAN ONE distinct '
          f'target label - i.e. identical inputs, different outcomes. This is a hard '
          f'floor on achievable accuracy: no model can resolve these.')
    return conflicting


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_data(path)
    quick_overview(df)

    target_col = infer_target(df)
    print(f'\nInferred target column: {target_col}')

    numeric_cols, cat_cols = infer_column_types(df)
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    print('\nNumeric cols:', numeric_cols)
    print('Categorical cols:', cat_cols)

    # ---- EDA plots (well over 5) ----
    plot_missingness(df)
    plot_distributions(df, numeric_cols)
    plot_boxplots(df, numeric_cols)
    plot_categorical_counts(df, cat_cols)

    numeric_corr = plot_correlation_heatmap(df, numeric_cols) if len(numeric_cols) >= 2 else None
    cat_corr = categorical_correlation_matrix(df, cat_cols) if len(cat_cols) >= 2 else None

    pair_cols = numeric_cols[:5]
    if len(pair_cols) >= 2:
        plot_pairplot(df, pair_cols, hue=target_col if df[target_col].nunique() <= 6 else None)

    # ---- Indicator feature analysis vs target ----
    task_is_binary = df[target_col].nunique(dropna=True) == 2
    task = 'classification' if task_is_binary else (
        'classification' if df[target_col].dtype == object else 'regression'
    )
    ranked_indicators = indicator_feature_analysis(df, numeric_cols, cat_cols, target_col, task)

    # ---- Encoder recommendations ----
    recommend_encoders(df, cat_cols)

    # ---- Linearity check ----
    linearity = simple_linearity_check(df, numeric_cols, target_col)

    # ---- Preprocessing for modeling ----
    df_model = df.copy()
    df_model = df_model.dropna(subset=[target_col])

    feature_cols_before_encoding = numeric_cols + cat_cols
    duplicate_conflict_check(df_model, feature_cols_before_encoding, target_col)

    for c in cat_cols:
        if df_model[c].nunique() <= 10:
            dummies = pd.get_dummies(df_model[c], prefix=c, dummy_na=True)
            df_model = pd.concat([df_model.drop(columns=[c]), dummies], axis=1)
        else:
            freq = df_model[c].value_counts(normalize=True)
            df_model[c + '_freq_enc'] = df_model[c].map(freq)
            df_model = df_model.drop(columns=[c])

    final_numeric = df_model.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in final_numeric:
        final_numeric.remove(target_col)

    X = df_model[final_numeric].fillna(df_model[final_numeric].median())
    y = df_model[target_col]

    task = 'classification' if y.nunique(dropna=True) == 2 else 'regression'
    print('\nDetermined task type:', task)

    # mutual information as a model-agnostic feature ranking (captures non-linear signal)
    mi_fn = mutual_info_classif if task == 'classification' else mutual_info_regression
    mi = mi_fn(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    print('\nTop features by mutual information with target (captures non-linear signal):')
    print(mi_series.head(15))
    plt.figure(figsize=(7, max(4, 0.3 * min(20, len(mi_series)))))
    mi_series.head(20).sort_values().plot(kind='barh')
    plt.title('Mutual information with target (top 20)')
    plt.savefig(os.path.join(PLOTS_DIR, 'mutual_information.png'), bbox_inches='tight')
    plt.close()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    analyze_bias_variance_noise(X_scaled, y, task=task)

    print('\nAll plots saved to ./plots/. Review the PNGs for visual EDA, target-association '
          'plots, learning curves, and CV-spread diagnostics.')


if __name__ == '__main__':
    main()