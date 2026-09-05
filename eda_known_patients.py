"""
EDA and preprocessing script for Known_Patients_01.csv
Generates >5 plots, inspects bias/variance/noise, recommends encoders, and checks linearity.
Run: python eda_known_patients.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, roc_auc_score

sns.set(style='whitegrid')

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


def infer_target(df):
    candidates = ['target', 'label', 'outcome', 'diagnosis', 'result']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: last column
    return df.columns[-1]


def load_data(path='Known_Patients_01.csv'):
    if not os.path.exists(path):
        print(f"Error: {path} not found in working directory.")
        sys.exit(1)
    df = pd.read_csv(path)
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


def plot_distributions(df, numeric_cols):
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        f = os.path.join(PLOTS_DIR, f'dist_{col}.png')
        plt.savefig(f, bbox_inches='tight')
        plt.close()


def plot_boxplots(df, numeric_cols):
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col].dropna())
        plt.title(f'Boxplot of {col}')
        f = os.path.join(PLOTS_DIR, f'box_{col}.png')
        plt.savefig(f, bbox_inches='tight')
        plt.close()


def plot_correlation_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation heatmap')
    f = os.path.join(PLOTS_DIR, 'corr_heatmap.png')
    plt.savefig(f, bbox_inches='tight')
    plt.close()
    return corr


def plot_pairplot(df, cols):
    sns.pairplot(df[cols].dropna(), corner=True)
    f = os.path.join(PLOTS_DIR, 'pairplot.png')
    plt.savefig(f, bbox_inches='tight')
    plt.close()


def plot_categorical_counts(df, cat_cols):
    for col in cat_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'Counts of {col}')
        f = os.path.join(PLOTS_DIR, f'count_{col}.png')
        plt.savefig(f, bbox_inches='tight')
        plt.close()


def scatter_with_reg(df, x, y):
    plt.figure(figsize=(6,4))
    sns.regplot(x=x, y=y, data=df, scatter_kws={'s':10}, line_kws={'color':'red'})
    plt.title(f'{y} vs {x} (with linear fit)')
    f = os.path.join(PLOTS_DIR, f'scatter_{y}_vs_{x}.png')
    plt.savefig(f, bbox_inches='tight')
    plt.close()


def infer_column_types(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # also treat low-cardinality numerics as categorical
    for col in numeric_cols[:]:
        if df[col].nunique() <= 10:
            obj_cols.append(col)
            numeric_cols.remove(col)
    return numeric_cols, obj_cols


def recommend_encoders(df, cat_cols):
    recommendations = {}
    for col in cat_cols:
        nunique = df[col].nunique(dropna=False)
        if nunique <= 10:
            recommendations[col] = 'OneHotEncoder (nominal, low cardinality)'
        else:
            recommendations[col] = 'Target encoding / OrdinalEncoder if ordered or frequency encoding'
    print('\nEncoder recommendations:')
    for k,v in recommendations.items():
        print(f'- {k}: {v}')
    return recommendations


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
    print('\nLinearity (R^2) by numeric predictor:')
    for k,v in sorted(results.items(), key=lambda x:-x[1])[:10]:
        print(f'- {k}: R^2={v:.3f}')
    return results


def analyze_bias_variance(df, X, y, task='regression'):
    print('\n=== Bias/Variance Investigation ===')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if task == 'regression':
        models = [LinearRegression(), RandomForestRegressor(n_estimators=50, random_state=42)]
        scorer = r2_score
    else:
        models = [LogisticRegression(max_iter=1000), RandomForestClassifier(n_estimators=50, random_state=42)]
        scorer = accuracy_score

    for m in models:
        mname = m.__class__.__name__
        m.fit(X_train, y_train)
        train_pred = m.predict(X_train)
        test_pred = m.predict(X_test)
        train_score = scorer(y_train, train_pred) if task!='regression' else r2_score(y_train, train_pred)
        test_score = scorer(y_test, test_pred) if task!='regression' else r2_score(y_test, test_pred)
        print(f'{mname}: train_score={train_score:.3f}, test_score={test_score:.3f}')

        # learning curve
        plt.figure(figsize=(6,4))
        train_sizes, train_scores, test_scores = learning_curve(m, X, y, cv=5, scoring='r2' if task=='regression' else 'accuracy', train_sizes=np.linspace(0.1,1.0,5))
        train_scores_mean = train_scores.mean(axis=1)
        test_scores_mean = test_scores.mean(axis=1)
        plt.plot(train_sizes, train_scores_mean, 'o-', label='train')
        plt.plot(train_sizes, test_scores_mean, 'o-', label='cv')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.title(f'Learning curve: {mname}')
        plt.legend()
        f = os.path.join(PLOTS_DIR, f'learning_curve_{mname}.png')
        plt.savefig(f, bbox_inches='tight')
        plt.close()


def main():
    df = load_data()
    quick_overview(df)

    target_col = infer_target(df)
    print(f'\nInferred target column: {target_col}')

    numeric_cols, cat_cols = infer_column_types(df)
    print('\nNumeric cols:', numeric_cols)
    print('Categorical cols:', cat_cols)

    # Ensure we don't accidentally encode or drop the target column during preprocessing
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    # EDA plots (>5)
    plot_distributions(df, numeric_cols[:6])
    plot_boxplots(df, numeric_cols[:6])
    plot_correlation_heatmap(df, numeric_cols)
    pair_cols = numeric_cols[:5] + ([target_col] if target_col not in numeric_cols else [])
    if len(pair_cols) >= 3:
        plot_pairplot(df, pair_cols)
    plot_categorical_counts(df, cat_cols[:3])

    # Scatter/regressions for top correlations
    corr = None
    if len(numeric_cols) >= 2:
        corr = plot_correlation_heatmap(df, numeric_cols)

    # Encoder recommendations
    recommend_encoders(df, cat_cols)

    # Linearity check
    linearity = simple_linearity_check(df, numeric_cols, target_col)

    # Prepare basic preprocessing for modeling
    df_model = df.copy()
    # simple strategy: drop rows with missing target
    df_model = df_model.dropna(subset=[target_col])

    # Encode categorical: one-hot for low-cardinality, label for high
    for c in cat_cols:
        if df_model[c].nunique() <= 10:
            dummies = pd.get_dummies(df_model[c], prefix=c, dummy_na=True)
            df_model = pd.concat([df_model.drop(columns=[c]), dummies], axis=1)
        else:
            # frequency encoding
            freq = df_model[c].value_counts(normalize=True)
            df_model[c+'_freq_enc'] = df_model[c].map(freq)
            df_model = df_model.drop(columns=[c])

    # final numeric features
    final_numeric = df_model.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in final_numeric:
        final_numeric.remove(target_col)

    X = df_model[final_numeric].fillna(df_model[final_numeric].median())
    y = df_model[target_col]

    # Determine task type
    task = 'regression' if pd.api.types.is_numeric_dtype(y) else 'classification'
    print('\nDetermined task type:', task)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Bias/variance analysis
    analyze_bias_variance(df_model, X_scaled, y, task=task)

    print('\nPlots saved to ./plots/. Review the PNG files for visual EDA and learning curves.')


if __name__ == '__main__':
    main()
