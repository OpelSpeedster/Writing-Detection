"""
src/preprocess.py
=================
Production-grade preprocessing module for the Kaggle
"AI vs Human Academic Writing Dataset" binary classification task.

Public API
----------
load_and_preprocess(csv_path, target_col='Is_AI_Assisted', test_size=0.2, random_state=42)
    -> (preprocessor_pipeline, X_train, X_test, y_train, y_test)

The returned ``preprocessor_pipeline`` is a fitted sklearn Pipeline:
    FunctionTransformer(feature_engineering) -> ColumnTransformer(impute+scale+encode)

It accepts raw feature DataFrames (no engineered columns needed at inference time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL = "Is_AI_Assisted"
LABEL_MAP = {"AI": 1, "Human": 0}

# Skewed telemetry columns that benefit from log1p compression
_LOG_SKEWED_COLS = ["Editing_Time", "Revision_Count", "Grammar_Errors", "Typing_Speed"]

# Ratio feature specs: (new_col_name, numerator_col, denominator_col)
# Denominator is evaluated as denominator_col + 1 to prevent zero-division.
_RATIO_SPECS = [
    ("words_per_minute", "Word_Count", "Editing_Time"),
    ("revision_density", "Revision_Count", "Word_Count"),
    ("lexical_diversity", "Unique_Words", "Total_Words"),
]

# ---------------------------------------------------------------------------
# Internal helpers (all pure functions on DataFrames)
# ---------------------------------------------------------------------------


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (mimics logging glitches)."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"[preprocess] Dropped {before - after} duplicate rows.")
    return df.reset_index(drop=True)


def _drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-predictive identifier columns (any column whose name contains 'id')."""
    id_cols = [c for c in df.columns if "id" in c.lower()]
    if id_cols:
        print(f"[preprocess] Dropping ID-like columns: {id_cols}")
        df = df.drop(columns=id_cols)
    return df


def _encode_label(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Map 'AI' -> 1, 'Human' -> 0 in the target column.

    No-op when the column is already numeric (e.g. Is_AI_Assisted is already 0/1 int).
    """
    if pd.api.types.is_numeric_dtype(df[target_col]):
        return df
    df = df.copy()
    df[target_col] = df[target_col].map(LABEL_MAP)
    unmapped = df[target_col].isna().sum()
    if unmapped:
        raise ValueError(
            f"[preprocess] {unmapped} rows have unmapped label values. "
            f"Expected 'AI' or 'Human'."
        )
    df[target_col] = df[target_col].astype(int)
    return df


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer rate/ratio features from raw counts.

    Uses (denominator + 1) to guard against zero-division.
    Replaces any resulting Inf / NaN with 0 after computation.
    """
    df = df.copy()
    for new_col, num_col, den_col in _RATIO_SPECS:
        if num_col in df.columns and den_col in df.columns:
            df[new_col] = df[num_col] / (df[den_col] + 1)
    # Guard: replace any inf or nan that may have slipped through
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply np.log1p to heavily skewed telemetry columns (if present).

    Clips values to >= 0 first so log1p never receives negative input.
    """
    df = df.copy()
    cols_to_transform = [c for c in _LOG_SKEWED_COLS if c in df.columns]
    for col in cols_to_transform:
        # Cast to float first so Python None → NaN (avoids object-dtype ufunc errors)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = np.log1p(df[col].clip(lower=0))
    return df


def _feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """Combined feature engineering step: ratios then log transforms.

    Wrapped in a FunctionTransformer so it becomes a stateless sklearn step.
    """
    X = _add_ratio_features(X)
    X = _log_transform(X)
    return X


def _build_preprocessor_pipeline(X_raw: pd.DataFrame) -> Pipeline:
    """
    Build a full sklearn Pipeline:
        step 1 — FunctionTransformer(_feature_engineering): ratio + log features
        step 2 — ColumnTransformer: median impute + RobustScaler (numeric)
                                     most-frequent impute + OneHotEncoder (categorical)

    X_raw must be passed so that column names after feature engineering can be inferred
    (needed to configure the ColumnTransformer correctly).
    """
    # Apply engineering once to discover the full post-engineering column set
    X_engineered = _feature_engineering(X_raw)

    numeric_cols = X_engineered.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_engineered.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: list[tuple] = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(_feature_engineering, validate=False),
            ),
            ("column_transformer", column_transformer),
        ]
    )


# Keep the old name as an alias so existing code that imports _build_preprocessor still works
def _build_preprocessor(X: pd.DataFrame) -> Pipeline:
    """Alias for _build_preprocessor_pipeline (backward-compat)."""
    return _build_preprocessor_pipeline(X)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_preprocess(
    csv_path: str,
    target_col: str = "Is_AI_Assisted",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the CSV, clean it, split, and return a fitted sklearn preprocessing
    Pipeline alongside stratified train/test splits of RAW features.

    The returned Pipeline is fully self-contained:
        raw DataFrame -> feature engineering -> impute/scale/encode -> numpy array

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (Kaggle "AI vs Human Academic Writing" dataset
        or any structurally compatible CSV).
    target_col : str, optional
        Name of the binary target column. Default: 'Is_AI_Assisted'.
        Pass 'label' for synthetic test fixtures that use string labels.
    test_size : float, optional
        Fraction of data to reserve for testing. Default: 0.2.
    random_state : int, optional
        Random seed for reproducibility. Default: 42.

    Returns
    -------
    preprocessor : sklearn.pipeline.Pipeline
        Fitted on X_train. Call ``preprocessor.transform(X_test)`` downstream.
        Accepts raw feature DataFrames — no pre-engineering needed at inference.
    X_train : pd.DataFrame  (raw features, no engineered cols)
    X_test  : pd.DataFrame  (raw features, no engineered cols)
    y_train : pd.Series  (binary int: AI=1, Human=0)
    y_test  : pd.Series  (binary int: AI=1, Human=0)
    """
    # 1. Load raw CSV
    print(f"[preprocess] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[preprocess] Raw shape: {df.shape}")

    # 2. Remove exact duplicate rows (logging glitches)
    df = _drop_duplicates(df)

    # 3. Drop non-predictive ID columns
    df = _drop_id_columns(df)

    # 4. Encode string label -> binary int (no-op if already numeric)
    df = _encode_label(df, target_col=target_col)

    # 5. Separate features and target (NO pre-split feature engineering)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"[preprocess] Class balance  : {y.value_counts().to_dict()}")

    # 6. Stratified train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(
        f"[preprocess] Split          : X_train={X_train.shape}, X_test={X_test.shape}"
    )

    # 7. Build and fit the full preprocessing Pipeline on training data only (no leakage)
    preprocessor = _build_preprocessor_pipeline(X_train)
    preprocessor.fit(X_train)

    return preprocessor, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    # Build a 50-row synthetic DataFrame for a quick end-to-end demo
    rng = np.random.default_rng(seed=42)
    n = 50
    demo_df = pd.DataFrame(
        {
            "submission_id": range(n),
            "Editing_Time": rng.exponential(scale=120, size=n),
            "Word_Count": rng.integers(200, 1500, size=n).astype(float),
            "Revision_Count": rng.integers(0, 50, size=n).astype(float),
            "Unique_Words": rng.integers(100, 800, size=n).astype(float),
            "Total_Words": rng.integers(200, 1500, size=n).astype(float),
            "Plagiarism_Score": rng.uniform(0, 1, size=n),
            "Keystroke_Speed": rng.uniform(20, 120, size=n),
            "essay_topic": rng.choice(["science", "history", "arts"], size=n),
            "Is_AI_Assisted": rng.integers(0, 2, size=n),
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        demo_df.to_csv(f, index=False)
        tmp_path = f.name

    try:
        preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess(
            tmp_path, target_col="Is_AI_Assisted", test_size=0.2, random_state=42
        )

        print("\n-- Results --------------------------------------------------")
        print(f"X_train shape (raw)     : {X_train.shape}")
        print(f"X_test shape  (raw)     : {X_test.shape}")
        print(f"y_train distribution    : {y_train.value_counts().to_dict()}")
        print(f"y_test  distribution    : {y_test.value_counts().to_dict()}")

        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        print(f"Transformed X_train     : {X_train_t.shape}")
        print(f"Transformed X_test      : {X_test_t.shape}")
        print(f"Any NaN in train output : {np.isnan(X_train_t).any()}")
        print(f"Any NaN in test  output : {np.isnan(X_test_t).any()}")
    finally:
        os.unlink(tmp_path)
