# src/preprocess.py — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade `src/preprocess.py` module that cleans, engineers features from, and produces a scikit-learn preprocessing pipeline for the "AI vs Human Academic Writing" Kaggle dataset.

**Architecture:** A single module (`src/preprocess.py`) exposing one primary public function — `load_and_preprocess()` — which returns a fitted `ColumnTransformer` preprocessor plus stratified train/test splits. All transformation logic (deduplication, ratio feature creation, log transforms, column detection, pipeline construction) lives inside this one file. A thin test file validates each logical unit with a synthetic DataFrame.

**Tech Stack:** Python 3.9+, pandas, numpy, scikit-learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `RobustScaler`, `OneHotEncoder`), pytest

## Global Constraints

- Python ≥ 3.9
- scikit-learn ≥ 1.2 (required for `sparse_output=False` on `OneHotEncoder`)
- No external data download; tests use a 30-row synthetic DataFrame
- Target column name: `label` (string `"AI"` / `"Human"`) → binary int (`1` / `0`)
- ID-like columns to drop: any column whose name contains `id` (case-insensitive)
- `test_size=0.2`, `random_state=42` as defaults
- All ratio/log transforms are applied **before** the sklearn pipeline (in pandas), then the resulting numeric frame is handed to `ColumnTransformer`

---

### Task 1: Project scaffold + dependency manifest

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)

**Interfaces:**
- Produces: installable dependency list consumed by all subsequent tasks

- [ ] **Step 1: Create `requirements.txt`**

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
pytest>=7.0
```

- [ ] **Step 2: Create empty init files**

```bash
# PowerShell
New-Item -ItemType File src/__init__.py
New-Item -ItemType File tests/__init__.py
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 4: Verify install**

```bash
python -c "import pandas, numpy, sklearn; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt src/__init__.py tests/__init__.py
git commit -m "chore: scaffold project and pin dependencies"
```

---

### Task 2: Write failing tests for `src/preprocess.py`

**Files:**
- Create: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: nothing yet (module does not exist; all tests must fail)
- Produces: `tests/test_preprocess.py` with full test coverage for every logical unit

- [ ] **Step 1: Write the test file**

```python
# tests/test_preprocess.py
"""
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
    assert not enriched.isnull().any().any(), "No NaN/Inf allowed after ratio features"
    assert not np.isinf(enriched.select_dtypes("number").values).any()


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
    before_max = df["Editing_Time"].max()
    transformed = _log_transform(df)
    after_max = transformed["Editing_Time"].max()
    assert after_max < before_max, "Log transform must reduce max of skewed feature"


# ── pipeline construction ──────────────────────────────────────────────────────

def test_build_preprocessor_returns_column_transformer():
    from src.preprocess import _build_preprocessor
    df = make_synthetic_df(n=30).drop_duplicates().copy()
    # Remove label & id cols before building preprocessor
    df = df[[c for c in df.columns if "id" not in c.lower() and c != "label"]]
    preprocessor = _build_preprocessor(df)
    assert isinstance(preprocessor, ColumnTransformer)


# ── end-to-end ────────────────────────────────────────────────────────────────

def test_load_and_preprocess_shapes(tmp_path):
    from src.preprocess import load_and_preprocess
    csv_path = tmp_path / "test_data.csv"
    make_synthetic_df(n=30).to_csv(csv_path, index=False)
    preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess(
        str(csv_path), test_size=0.2, random_state=42
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
        str(csv_path), test_size=0.2, random_state=42
    )
    # Stratified split: both sets should contain both classes
    assert 0 in y_train.values and 1 in y_train.values
    assert 0 in y_test.values and 1 in y_test.values


def test_load_and_preprocess_returns_column_transformer(tmp_path):
    from src.preprocess import load_and_preprocess
    csv_path = tmp_path / "test_data.csv"
    make_synthetic_df(n=30).to_csv(csv_path, index=False)
    preprocessor, *_ = load_and_preprocess(str(csv_path))
    assert isinstance(preprocessor, ColumnTransformer)


def test_no_nan_in_transformed_output(tmp_path):
    from src.preprocess import load_and_preprocess
    import numpy as np
    csv_path = tmp_path / "test_data.csv"
    make_synthetic_df(n=30).to_csv(csv_path, index=False)
    preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess(
        str(csv_path)
    )
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    assert not np.isnan(X_train_t).any(), "No NaN in transformed train set"
    assert not np.isnan(X_test_t).any(), "No NaN in transformed test set"
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
pytest tests/test_preprocess.py -v
```
Expected: All tests `FAILED` or `ERROR` with `ModuleNotFoundError: No module named 'src.preprocess'`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_preprocess.py
git commit -m "test: add failing tests for preprocess module"
```

---

### Task 3: Implement `src/preprocess.py`

**Files:**
- Create: `src/preprocess.py`

**Interfaces:**
- Consumes: CSV file at `csv_path`; columns `Editing_Time`, `Word_Count`, `Revision_Count`, `Unique_Words`, `Total_Words`, `label`; any categorical object columns; any `*id*` columns to drop
- Produces:
  - `_drop_duplicates(df: pd.DataFrame) -> pd.DataFrame`
  - `_drop_id_columns(df: pd.DataFrame) -> pd.DataFrame`
  - `_encode_label(df: pd.DataFrame, target_col: str = "label") -> pd.DataFrame`
  - `_add_ratio_features(df: pd.DataFrame) -> pd.DataFrame`
  - `_log_transform(df: pd.DataFrame) -> pd.DataFrame`
  - `_build_preprocessor(X: pd.DataFrame) -> ColumnTransformer`
  - `load_and_preprocess(csv_path: str, test_size: float = 0.2, random_state: int = 42) -> tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]`

- [ ] **Step 1: Write `src/preprocess.py`**

```python
"""
src/preprocess.py
=================
Production-grade preprocessing module for the Kaggle
"AI vs Human Academic Writing Dataset" binary classification task.

Public API
----------
load_and_preprocess(csv_path, test_size=0.2, random_state=42)
    -> (preprocessor, X_train, X_test, y_train, y_test)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL = "label"
LABEL_MAP = {"AI": 1, "Human": 0}

# Skewed telemetry columns that benefit from log1p compression
_LOG_SKEWED_COLS = ["Editing_Time", "Revision_Count"]

# Ratio features to engineer; tuples of (new_col, numerator, denominator_expr)
# Denominator expressions are evaluated safely with +1 to avoid zero-division.
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
    """Map 'AI' -> 1, 'Human' -> 0 in the target column."""
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
    Uses denominator + 1 to guard against zero-division.
    Replaces any resulting Inf/NaN with 0.
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
    """Apply np.log1p to heavily skewed telemetry columns (if present)."""
    df = df.copy()
    cols_to_transform = [c for c in _LOG_SKEWED_COLS if c in df.columns]
    for col in cols_to_transform:
        df[col] = np.log1p(df[col].clip(lower=0))
    return df


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - For numeric columns: median impute → RobustScaler
    - For categorical columns: most-frequent impute → OneHotEncoder
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

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

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_preprocess(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the CSV, clean it, engineer features, split, and return a
    fitted ColumnTransformer preprocessor alongside train/test splits.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (Kaggle dataset or any compatible CSV).
    test_size : float, optional
        Fraction of data to reserve for testing (default 0.2).
    random_state : int, optional
        Seed for reproducibility (default 42).

    Returns
    -------
    preprocessor : ColumnTransformer
        Fitted on X_train; ready to call .transform() on X_test.
    X_train : pd.DataFrame
    X_test  : pd.DataFrame
    y_train : pd.Series
    y_test  : pd.Series
    """
    # 1. Load
    print(f"[preprocess] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[preprocess] Raw shape: {df.shape}")

    # 2. Deduplicate
    df = _drop_duplicates(df)

    # 3. Remove IDs
    df = _drop_id_columns(df)

    # 4. Encode label
    df = _encode_label(df, target_col=TARGET_COL)

    # 5. Feature engineering (ratios, then log transforms)
    df = _add_ratio_features(df)
    df = _log_transform(df)

    # 6. Split features / target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"[preprocess] Class balance → {y.value_counts().to_dict()}")

    # 7. Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(
        f"[preprocess] Split: X_train={X_train.shape}, X_test={X_test.shape}"
    )

    # 8. Build and fit preprocessor on training data only
    preprocessor = _build_preprocessor(X_train)
    preprocessor.fit(X_train)

    return preprocessor, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    # Build a 50-row synthetic DataFrame for a quick demo
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
            "label": rng.choice(["AI", "Human"], size=n),
        }
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        demo_df.to_csv(f, index=False)
        tmp_path = f.name

    try:
        preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess(
            tmp_path, test_size=0.2, random_state=42
        )
        print("\n── Results ──────────────────────────────────────")
        print(f"X_train shape : {X_train.shape}")
        print(f"X_test shape  : {X_test.shape}")
        print(f"y_train dist  : {y_train.value_counts().to_dict()}")
        print(f"y_test dist   : {y_test.value_counts().to_dict()}")
        X_train_t = preprocessor.transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        print(f"Transformed X_train shape : {X_train_t.shape}")
        print(f"Transformed X_test shape  : {X_test_t.shape}")
        print("Any NaN in transformed train?", np.isnan(X_train_t).any())
        print("Any NaN in transformed test? ", np.isnan(X_test_t).any())
    finally:
        os.unlink(tmp_path)
```

- [ ] **Step 2: Run the tests**

```bash
pytest tests/test_preprocess.py -v
```
Expected: All tests `PASSED`

- [ ] **Step 3: Run the demo smoke-test**

```bash
python -m src.preprocess
```
Expected: printed shape/balance info, `Any NaN …` lines both `False`

- [ ] **Step 4: Commit**

```bash
git add src/preprocess.py
git commit -m "feat: implement load_and_preprocess module with TDD"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|---|---|
| Remove exact duplicates | Task 3 → `_drop_duplicates` |
| Drop non-predictive identifiers (submission/student ID) | Task 3 → `_drop_id_columns` |
| Map AI/Human → 1/0 | Task 3 → `_encode_label` |
| Words-per-minute ratio | Task 3 → `_add_ratio_features` |
| Revision density | Task 3 → `_add_ratio_features` |
| Lexical diversity | Task 3 → `_add_ratio_features` |
| log1p on skewed telemetry | Task 3 → `_log_transform` |
| Separate numeric / categorical columns | Task 3 → `_build_preprocessor` |
| Median impute numerics | Task 3 → `numeric_pipeline` |
| RobustScaler numerics | Task 3 → `numeric_pipeline` |
| Most-frequent impute categoricals | Task 3 → `categorical_pipeline` |
| OneHotEncoder(handle_unknown='ignore', sparse_output=False) | Task 3 → `categorical_pipeline` |
| ColumnTransformer | Task 3 → `_build_preprocessor` |
| `load_and_preprocess(csv_path, test_size, random_state)` signature | Task 3 |
| Returns `(preprocessor, X_train, X_test, y_train, y_test)` | Task 3 |
| Stratified split | Task 3 |
| `if __name__ == '__main__':` demo block | Task 3 |

All requirements covered. ✓

### Placeholder scan

No TBD, TODO, or incomplete sections found.

### Type consistency

All private helper function names and signatures used in `tests/test_preprocess.py` match exactly those defined in the `src/preprocess.py` implementation.
