"""
tests/test_preprocess.py
========================
Tests for src/preprocess.py — AI vs Human Academic Writing preprocessing module.
All tests use a 30-row synthetic DataFrame that mimics the Kaggle dataset structure.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer


# ── helpers ──────────────────────────────────────────────────────────────────

def make_synthetic_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """30-row DataFrame that mimics the Kaggle dataset."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "submission_id": range(n),
            "student_id": range(1000, 1000 + n),
            "Editing_Time": rng.exponential(scale=120, size=n),
            "Word_Count": rng.integers(200, 1500, size=n).astype(float),
            "Revision_Count": rng.integers(0, 50, size=n).astype(float),
            "Unique_Words": rng.integers(100, 800, size=n).astype(float),
            "Total_Words": rng.integers(200, 1500, size=n).astype(float),
            "Plagiarism_Score": rng.uniform(0, 1, size=n),
            "Keystroke_Speed": rng.uniform(20, 120, size=n),
            "essay_topic": rng.choice(["science", "history", "arts"], size=n),
            "label": rng.choice(["AI", "Human"], size=n),
        }
    )
    # Introduce 3 exact duplicates (logging glitch simulation)
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    # Introduce some missing values
    df.loc[5, "Editing_Time"] = np.nan
    df.loc[10, "Revision_Count"] = np.nan
    df.loc[15, "essay_topic"] = np.nan
    return df


# ── import guard ──────────────────────────────────────────────────────────────

def test_module_importable():
    """The module must be importable without errors."""
    import src.preprocess  # noqa: F401


# ── deduplication ─────────────────────────────────────────────────────────────

def test_drop_duplicates_removes_exact_clones():
    from src.preprocess import _drop_duplicates
    df = make_synthetic_df(n=30)
    original_len = 30 + 3  # 33 rows including duplicates
    cleaned = _drop_duplicates(df)
    assert len(cleaned) == 30, f"Expected 30 rows after dedup, got {len(cleaned)}"


# ── ID column removal ─────────────────────────────────────────────────────────

def test_drop_id_columns_removes_id_like_cols():
    from src.preprocess import _drop_id_columns
    df = make_synthetic_df(n=30).drop_duplicates()
    cleaned = _drop_id_columns(df)
    id_cols = [c for c in cleaned.columns if "id" in c.lower()]
    assert len(id_cols) == 0, f"Expected no ID columns, found: {id_cols}"


# ── label encoding ────────────────────────────────────────────────────────────

def test_encode_label_produces_binary_ints():
    from src.preprocess import _encode_label
    df = pd.DataFrame({"label": ["AI", "Human", "AI", "Human"]})
    result = _encode_label(df, target_col="label")
    assert set(result["label"].unique()).issubset(
        {0, 1}
    ), "Label must be binary 0/1 integers"
    assert result.loc[result["label"] == 1, :].shape[0] == 2  # AI rows


# ── feature engineering ───────────────────────────────────────────────────────

def test_add_ratio_features_no_divide_by_zero():
    from src.preprocess import _add_ratio_features
    df = make_synthetic_df(n=30).drop_duplicates().copy()
    df["Word_Count"] = 0  # edge-case: all zeros
    enriched = _add_ratio_features(df)
    # Only numeric columns must be free of NaN/Inf; categorical NaN is a separate concern
    numeric_vals = enriched.select_dtypes("number").values
    assert not np.isnan(numeric_vals).any(), "No NaN allowed in numeric cols after ratio features"
    assert not np.isinf(numeric_vals).any(), "No Inf allowed in numeric cols after ratio features"


def test_add_ratio_features_column_presence():
    from src.preprocess import _add_ratio_features
    df = make_synthetic_df(n=30).drop_duplicates().copy()
    enriched = _add_ratio_features(df)
    assert "words_per_minute" in enriched.columns
    assert "revision_density" in enriched.columns
    assert "lexical_diversity" in enriched.columns


def test_log_transform_skewed_features():
    from src.preprocess import _log_transform
    df = make_synthetic_df(n=30).drop_duplicates().copy()
    before_max = df["Editing_Time"].dropna().max()
    transformed = _log_transform(df)
    after_max = transformed["Editing_Time"].max()
    assert after_max < before_max, "Log transform must reduce max of skewed feature"


# ── pipeline construction ──────────────────────────────────────────────────────

def test_build_preprocessor_returns_pipeline():
    from src.preprocess import _build_preprocessor
    from sklearn.pipeline import Pipeline
    df = make_synthetic_df(n=30).drop_duplicates().copy()
    # Remove label & id cols before building preprocessor
    df = df[[c for c in df.columns if "id" not in c.lower() and c != "label"]]
    preprocessor = _build_preprocessor(df)
    assert isinstance(preprocessor, Pipeline), (
        f"Expected Pipeline, got {type(preprocessor)}"
    )


# ── end-to-end ────────────────────────────────────────────────────────────────

def test_load_and_preprocess_shapes(tmp_path):
    from src.preprocess import load_and_preprocess
    csv_path = tmp_path / "test_data.csv"
    make_synthetic_df(n=30).to_csv(csv_path, index=False)
    preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess(
        str(csv_path), target_col="label", test_size=0.2, random_state=42
    )
    # 33 rows total, 3 dupes removed → 30 rows → 24 train / 6 test
    assert X_train.shape[0] == 24
    assert X_test.shape[0] == 6
    assert len(y_train) == 24
    assert len(y_test) == 6


def test_load_and_preprocess_label_balance(tmp_path):
    from src.preprocess import load_and_preprocess
    csv_path = tmp_path / "test_data.csv"
    make_synthetic_df(n=30).to_csv(csv_path, index=False)
    _, _, _, y_train, y_test = load_and_preprocess(
        str(csv_path), target_col="label", test_size=0.2, random_state=42
    )
    # Stratified split: both sets should contain both classes
    assert 0 in y_train.values and 1 in y_train.values
    assert 0 in y_test.values and 1 in y_test.values


def test_load_and_preprocess_returns_pipeline(tmp_path):
    from src.preprocess import load_and_preprocess
    from sklearn.pipeline import Pipeline
    csv_path = tmp_path / "test_data.csv"
    make_synthetic_df(n=30).to_csv(csv_path, index=False)
    preprocessor, *_ = load_and_preprocess(str(csv_path), target_col="label")
    assert isinstance(preprocessor, Pipeline), (
        f"Expected Pipeline, got {type(preprocessor)}"
    )


def test_no_nan_in_transformed_output(tmp_path):
    from src.preprocess import load_and_preprocess
    csv_path = tmp_path / "test_data.csv"
    make_synthetic_df(n=30).to_csv(csv_path, index=False)
    preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess(
        str(csv_path), target_col="label"
    )
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    assert not np.isnan(X_train_t).any(), "No NaN in transformed train set"
    assert not np.isnan(X_test_t).any(), "No NaN in transformed test set"
