"""
Data Preprocessing Module
=========================
Handles the credit risk dataset preprocessing pipeline, implementing the
transformations described in:

    "Quantum-Enhanced Smart Computing Framework for Sustainable Credit Risk
     Decision Communication" (Jain, Singh, et al., 2026)

Equations implemented:
    Eq. 1 — Mean Imputation
    Eq. 2 — Min-Max Normalisation to [0, 1]
    Eq. 3 — One-Hot Encoding
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a credit risk CSV dataset.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file (e.g., Kaggle *Credit Risk Dataset*).

    Returns
    -------
    pd.DataFrame
        Raw DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Imputation  (Eq. 1)
# ---------------------------------------------------------------------------

def mean_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing values in numeric columns with their column mean.

    Implements **Eq. 1** from the paper:

        x̂_j = (1 / |S_j|) Σ_{i ∈ S_j} x_{ij}

    where S_j is the set of non-missing indices for feature j.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (may contain NaN values).

    Returns
    -------
    pd.DataFrame
        Copy with NaN values in numeric columns filled by their means.
    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_out[col].isna().any():
            col_mean = df_out[col].mean()
            df_out[col] = df_out[col].fillna(col_mean)
    return df_out


# ---------------------------------------------------------------------------
# Normalisation  (Eq. 2)
# ---------------------------------------------------------------------------

def min_max_normalize(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Scale numeric features to the [0, 1] interval (quantum-compatible).

    Implements **Eq. 2** from the paper:

        x'_{ij} = (x_{ij} - min_j) / (max_j - min_j)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with imputed values.
    columns : list of str, optional
        Columns to normalise.  If *None*, all numeric columns are scaled.

    Returns
    -------
    pd.DataFrame
        Copy with specified columns scaled to [0, 1].
    """
    df_out = df.copy()
    if columns is None:
        columns = list(df_out.select_dtypes(include=[np.number]).columns)

    for col in columns:
        col_min = df_out[col].min()
        col_max = df_out[col].max()
        denom = col_max - col_min
        if denom == 0:
            df_out[col] = 0.0
        else:
            df_out[col] = (df_out[col] - col_min) / denom

    return df_out


# ---------------------------------------------------------------------------
# One-Hot Encoding  (Eq. 3)
# ---------------------------------------------------------------------------

def one_hot_encode(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply one-hot encoding to categorical columns.

    Implements **Eq. 3** from the paper:

        e_k(c) = [0, …, 1, …, 0]  (indicator vector for category c in feature k)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Categorical columns to encode.  If *None*, all ``object`` / ``category``
        dtype columns are encoded.

    Returns
    -------
    pd.DataFrame
        DataFrame with categorical columns replaced by one-hot columns.
    """
    df_out = df.copy()
    if columns is None:
        columns = list(df_out.select_dtypes(include=["object", "category"]).columns)

    df_out = pd.get_dummies(df_out, columns=columns, drop_first=False, dtype=float)
    return df_out


# ---------------------------------------------------------------------------
# Synthetic data generator (fallback when Kaggle CSV is unavailable)
# ---------------------------------------------------------------------------

def generate_synthetic_credit_data(
    n_samples: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic credit risk dataset for development and testing.

    The synthetic data mimics the structure of the Kaggle Credit Risk Dataset
    with 12 features and a binary target (``loan_status``).

    Parameters
    ----------
    n_samples : int, default 2000
        Number of samples to generate.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic credit risk DataFrame.
    """
    rng = np.random.default_rng(random_state)

    data = pd.DataFrame({
        "person_age": rng.integers(20, 70, n_samples).astype(float),
        "person_income": rng.lognormal(10.5, 0.8, n_samples),
        "person_emp_length": rng.exponential(5, n_samples).clip(0, 40),
        "loan_amnt": rng.lognormal(9, 0.7, n_samples),
        "loan_int_rate": rng.uniform(5.0, 25.0, n_samples),
        "loan_percent_income": rng.uniform(0.0, 0.8, n_samples),
        "cb_person_cred_hist_length": rng.integers(2, 30, n_samples).astype(float),
        "person_home_ownership": rng.choice(
            ["RENT", "OWN", "MORTGAGE", "OTHER"], n_samples, p=[0.4, 0.15, 0.4, 0.05]
        ),
        "loan_intent": rng.choice(
            ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
            n_samples,
        ),
        "loan_grade": rng.choice(["A", "B", "C", "D", "E", "F", "G"], n_samples),
        "cb_person_default_on_file": rng.choice(["Y", "N"], n_samples, p=[0.2, 0.8]),
    })

    # --- Synthetic target with realistic correlations ---
    risk_score = (
        0.3 * (data["loan_int_rate"] / 25)
        + 0.25 * data["loan_percent_income"]
        - 0.15 * (data["cb_person_cred_hist_length"] / 30)
        + 0.2 * (data["cb_person_default_on_file"] == "Y").astype(float)
        + 0.1 * rng.standard_normal(n_samples)
    )
    data["loan_status"] = (risk_score > np.median(risk_score)).astype(int)

    # Inject ~5 % missing values in numeric columns
    for col in ["person_emp_length", "loan_int_rate"]:
        mask = rng.random(n_samples) < 0.05
        data.loc[mask, col] = np.nan

    return data


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(
    path: Optional[str | Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    target_column: str = "loan_status",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Execute the complete preprocessing pipeline.

    Steps:
        1. Load dataset (or generate synthetic fallback).
        2. Mean imputation  (Eq. 1).
        3. One-hot encoding (Eq. 3).
        4. Min-max normalisation to [0, 1] (Eq. 2).
        5. Stratified train/test split (80/20).

    Parameters
    ----------
    path : str or Path, optional
        Path to CSV.  If *None* or file not found, synthetic data is used.
    test_size : float, default 0.2
        Fraction of data for the test set.
    random_state : int, default 42
        Random seed.
    target_column : str, default "loan_status"
        Name of the binary target column.

    Returns
    -------
    X_train, X_test : np.ndarray
        Feature matrices (normalised to [0, 1]).
    y_train, y_test : np.ndarray
        Binary label vectors.
    feature_names : list of str
        Column names after encoding.
    """
    # 1. Load
    if path is not None and Path(path).exists():
        df = load_dataset(path)
        print(f"[preprocessing] Loaded dataset from {path}  ({len(df)} rows)")
    else:
        df = generate_synthetic_credit_data(random_state=random_state)
        print(f"[preprocessing] Using synthetic data  ({len(df)} rows)")

    # 2. Imputation (Eq. 1)
    df = mean_imputation(df)

    # 3. One-hot encoding (Eq. 3)
    df = one_hot_encode(df)

    # 4. Separate features / target
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found.")
    y = df[target_column].values.astype(int)
    X_df = df.drop(columns=[target_column])

    # 5. Min-max normalisation (Eq. 2) — only numeric columns
    X_df = min_max_normalize(X_df)
    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float64)

    # 6. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    print(
        f"[preprocessing] Split: train={len(X_train)}, test={len(X_test)}, "
        f"features={X_train.shape[1]}"
    )

    return X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Preprocessing Module — Self-Test")
    print("=" * 60)

    X_tr, X_te, y_tr, y_te, feat_names = preprocess_pipeline()
    print(f"\nFeature names ({len(feat_names)}): {feat_names[:10]} ...")
    print(f"X_train range: [{X_tr.min():.4f}, {X_tr.max():.4f}]")
    print(f"y distribution (train): {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    print("\n✓ Preprocessing self-test passed.")
