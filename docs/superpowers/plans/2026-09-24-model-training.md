# Model Training — Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `src/train.py` — a module that loads the real Kaggle dataset, runs it through the updated preprocessor, trains a single XGBoost classifier (with `scale_pos_weight` for class imbalance), evaluates it with stratified 5-fold CV + held-out test metrics, and saves a ready-for-inference `models/pipeline.pkl`.

**Architecture:** Two source files change:
1. `src/preprocess.py` — patched for the real dataset: target col `Is_AI_Assisted` (already binary int, no string mapping), two extra log-transform columns (`Grammar_Errors`, `Typing_Speed`), and `Student_ID` dropped by the existing `_drop_id_columns` heuristic.
2. `src/train.py` — new module. Builds a `sklearn.pipeline.Pipeline(preprocessor → XGBClassifier)`, runs stratified 5-fold CV to report ROC-AUC / F1 / precision / recall, fits the full pipeline on `X_train`, evaluates on `X_test`, and saves to `models/pipeline.pkl` via joblib.

**Tech Stack:** Python 3.9+, pandas, numpy, scikit-learn ≥ 1.2, xgboost ≥ 1.7, joblib

## Global Constraints

- Target column: `Is_AI_Assisted` (int 0/1 — no string-to-int mapping needed)
- `scale_pos_weight` = `sum(y==0) / sum(y==1)` computed at fit-time from `y_train`
- Evaluation metrics reported: ROC-AUC, F1 (macro), Precision (macro), Recall (macro), Accuracy
- Artifact saved to `models/pipeline.pkl` (directory created if absent)
- `test_size=0.2`, `random_state=42` throughout
- No hyperparameter tuning — fixed well-chosen XGBoost defaults (see Task 3)
- All tests use a 100-row synthetic DataFrame with `Is_AI_Assisted` column; no real CSV required

---

### Task 1: Add xgboost + joblib to requirements; patch preprocess.py

**Files:**
- Modify: `requirements.txt`
- Modify: `src/preprocess.py`

**Interfaces:**
- Produces: `load_and_preprocess(csv_path, target_col='Is_AI_Assisted', test_size=0.2, random_state=42)` — `target_col` becomes a parameter with `Is_AI_Assisted` as its new default; function skips the `_encode_label` step when target is already binary int.

**Changes to `src/preprocess.py`:**

1. Add `target_col: str = "Is_AI_Assisted"` parameter to `load_and_preprocess`.
2. Expand `_LOG_SKEWED_COLS` to also include `"Grammar_Errors"` and `"Typing_Speed"`.
3. Make `_encode_label` a no-op when the column is already numeric (dtype is int/float).
4. Update the `__main__` demo block to use the new `target_col` default and a column named `Is_AI_Assisted`.

- [ ] **Step 1: Add deps to `requirements.txt`**

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
xgboost>=1.7
joblib>=1.2
pytest>=7.0
```

- [ ] **Step 2: Patch `_LOG_SKEWED_COLS` constant in `src/preprocess.py`**

Find:
```python
_LOG_SKEWED_COLS = ["Editing_Time", "Revision_Count"]
```
Replace with:
```python
_LOG_SKEWED_COLS = ["Editing_Time", "Revision_Count", "Grammar_Errors", "Typing_Speed"]
```

- [ ] **Step 3: Make `_encode_label` a no-op for already-binary columns**

Find the body of `_encode_label` and add an early-return guard:
```python
def _encode_label(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Map string labels to binary int. No-op when column is already numeric."""
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
```

- [ ] **Step 4: Add `target_col` parameter to `load_and_preprocess`**

Change signature from:
```python
def load_and_preprocess(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
```
to:
```python
def load_and_preprocess(
    csv_path: str,
    target_col: str = "Is_AI_Assisted",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
```

And replace all `TARGET_COL` usages inside the function body with `target_col`:
```python
    df = _encode_label(df, target_col=target_col)
    ...
    X = df.drop(columns=[target_col])
    y = df[target_col]
```

- [ ] **Step 5: Update `__main__` demo to use `Is_AI_Assisted`**

In the demo block, change:
```python
"label": rng.choice(["AI", "Human"], size=n),
```
to:
```python
"Is_AI_Assisted": rng.integers(0, 2, size=n),
```
And remove `target_col` kwarg from the `load_and_preprocess` call (default is now correct).

- [ ] **Step 6: Install new deps**

```bash
pip install xgboost>=1.7 joblib>=1.2 -q
```

- [ ] **Step 7: Run existing preprocess tests — must all pass**

```bash
pytest tests/test_preprocess.py -v
```
Expected: 12 passed (the synthetic tests still use `label` with string values, so the string-path of `_encode_label` remains exercised; no breakage expected because `target_col` defaults changed only for the public function, not for the internal helper).

---

### Task 2: Write failing tests for `src/train.py`

**Files:**
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: `src.train.train_and_evaluate(csv_path, target_col, test_size, random_state) -> dict`
- Consumes: `src.train.load_pipeline(path) -> sklearn.pipeline.Pipeline`
- Produces: test file covering import, return types, metric keys, artifact save/load round-trip

- [ ] **Step 1: Write the test file**

```python
# tests/test_train.py
"""Tests for src/train.py — XGBoost training pipeline."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ── synthetic data factory ────────────────────────────────────────────────────

def make_train_csv(tmp_path: Path, n: int = 100, seed: int = 0) -> str:
    """Write a synthetic CSV mimicking the real Kaggle dataset structure."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "Student_ID": [f"id{i:04d}" for i in range(n)],
            "Submission_Type": rng.choice(["Essay", "Lab_Report", "Research_Paper"], n),
            "Academic_Level": rng.choice(["Undergraduate", "Postgraduate", "High_School"], n),
            "Primary_Language": rng.choice(["Native", "Non_Native"], n),
            "Word_Count": rng.integers(200, 1500, n),
            "Average_Sentence_Length": rng.uniform(10, 30, n),
            "Grammar_Errors": rng.integers(0, 20, n).astype(float),
            "Vocabulary_Richness": rng.uniform(0.3, 0.9, n),
            "Passive_Voice_Ratio": rng.uniform(0, 0.5, n),
            "Reading_Level": rng.uniform(8, 16, n),
            "Editing_Time": rng.exponential(120, n),
            "Revision_Count": rng.integers(0, 30, n).astype(float),
            "Typing_Speed": rng.exponential(40, n),
            "Plagiarism_Score": rng.uniform(0, 5, n),
            "Citation_Count": rng.integers(0, 20, n).astype(float),
            "Punctuation_Density": rng.uniform(0.04, 0.12, n),
            "Sentence_Complexity": rng.uniform(1, 3, n),
            "Font_Family": rng.choice(["Arial", "Helvetica", "Calibri"], n),
            "Submission_Hour": rng.integers(0, 24, n),
            "Is_AI_Assisted": rng.choice([0, 1], n, p=[0.85, 0.15]),
        }
    )
    csv_path = tmp_path / "train_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


# ── import guard ──────────────────────────────────────────────────────────────

def test_train_module_importable():
    import src.train  # noqa: F401


# ── train_and_evaluate ────────────────────────────────────────────────────────

def test_train_and_evaluate_returns_dict(tmp_path):
    from src.train import train_and_evaluate
    csv_path = make_train_csv(tmp_path, n=100)
    result = train_and_evaluate(csv_path)
    assert isinstance(result, dict)


def test_train_and_evaluate_metric_keys(tmp_path):
    from src.train import train_and_evaluate
    csv_path = make_train_csv(tmp_path, n=100)
    result = train_and_evaluate(csv_path)
    expected_keys = {"roc_auc", "f1", "precision", "recall", "accuracy"}
    assert expected_keys.issubset(result.keys()), (
        f"Missing keys: {expected_keys - result.keys()}"
    )


def test_train_and_evaluate_metric_ranges(tmp_path):
    from src.train import train_and_evaluate
    csv_path = make_train_csv(tmp_path, n=100)
    result = train_and_evaluate(csv_path)
    for key, val in result.items():
        assert 0.0 <= val <= 1.0, f"Metric {key}={val} out of [0, 1]"


def test_train_and_evaluate_saves_pipeline(tmp_path):
    from src.train import train_and_evaluate
    csv_path = make_train_csv(tmp_path, n=100)
    model_path = str(tmp_path / "pipeline.pkl")
    train_and_evaluate(csv_path, model_path=model_path)
    assert Path(model_path).exists(), "pipeline.pkl was not created"


# ── load_pipeline ─────────────────────────────────────────────────────────────

def test_load_pipeline_returns_sklearn_pipeline(tmp_path):
    from src.train import train_and_evaluate, load_pipeline
    from sklearn.pipeline import Pipeline
    csv_path = make_train_csv(tmp_path, n=100)
    model_path = str(tmp_path / "pipeline.pkl")
    train_and_evaluate(csv_path, model_path=model_path)
    pipeline = load_pipeline(model_path)
    assert isinstance(pipeline, Pipeline)


def test_load_pipeline_can_predict(tmp_path):
    from src.train import train_and_evaluate, load_pipeline
    csv_path = make_train_csv(tmp_path, n=100)
    model_path = str(tmp_path / "pipeline.pkl")
    train_and_evaluate(csv_path, model_path=model_path)
    pipeline = load_pipeline(model_path)

    # Build a minimal single-row DataFrame matching the schema
    rng = np.random.default_rng(99)
    sample = pd.DataFrame(
        {
            "Student_ID": ["testid"],
            "Submission_Type": ["Essay"],
            "Academic_Level": ["Undergraduate"],
            "Primary_Language": ["Native"],
            "Word_Count": [800],
            "Average_Sentence_Length": [18.0],
            "Grammar_Errors": [3.0],
            "Vocabulary_Richness": [0.6],
            "Passive_Voice_Ratio": [0.2],
            "Reading_Level": [12.0],
            "Editing_Time": [90.0],
            "Revision_Count": [5.0],
            "Typing_Speed": [45.0],
            "Plagiarism_Score": [1.5],
            "Citation_Count": [4.0],
            "Punctuation_Density": [0.07],
            "Sentence_Complexity": [2.0],
            "Font_Family": ["Arial"],
            "Submission_Hour": [14],
        }
    )
    preds = pipeline.predict(sample)
    assert preds.shape == (1,)
    assert preds[0] in (0, 1)


def test_load_pipeline_predict_proba(tmp_path):
    from src.train import train_and_evaluate, load_pipeline
    csv_path = make_train_csv(tmp_path, n=100)
    model_path = str(tmp_path / "pipeline.pkl")
    train_and_evaluate(csv_path, model_path=model_path)
    pipeline = load_pipeline(model_path)

    sample = pd.DataFrame(
        {
            "Student_ID": ["testid"],
            "Submission_Type": ["Essay"],
            "Academic_Level": ["Undergraduate"],
            "Primary_Language": ["Native"],
            "Word_Count": [800],
            "Average_Sentence_Length": [18.0],
            "Grammar_Errors": [3.0],
            "Vocabulary_Richness": [0.6],
            "Passive_Voice_Ratio": [0.2],
            "Reading_Level": [12.0],
            "Editing_Time": [90.0],
            "Revision_Count": [5.0],
            "Typing_Speed": [45.0],
            "Plagiarism_Score": [1.5],
            "Citation_Count": [4.0],
            "Punctuation_Density": [0.07],
            "Sentence_Complexity": [2.0],
            "Font_Family": ["Arial"],
            "Submission_Hour": [14],
        }
    )
    proba = pipeline.predict_proba(sample)
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests to confirm they all fail**

```bash
pytest tests/test_train.py -v
```
Expected: All `FAILED` or `ERROR` with `ModuleNotFoundError: No module named 'src.train'`

---

### Task 3: Implement `src/train.py`

**Files:**
- Create: `src/train.py`
- Create: `models/` directory (created at runtime by the module)

**Interfaces:**
- Consumes: `src.preprocess.load_and_preprocess(csv_path, target_col, test_size, random_state)`
- Produces:
  - `train_and_evaluate(csv_path, target_col='Is_AI_Assisted', test_size=0.2, random_state=42, model_path='models/pipeline.pkl') -> dict`
  - `load_pipeline(path: str) -> sklearn.pipeline.Pipeline`

**Fixed XGBoost hyperparameters (production-ready defaults for tabular imbalanced data):**
```python
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}
```
`scale_pos_weight` is NOT in this dict — it is computed dynamically from `y_train` at fit-time.

- [ ] **Step 1: Write `src/train.py`**

```python
"""
src/train.py
============
Train an XGBoost binary classifier on the AI vs Human Academic Writing dataset.

Public API
----------
train_and_evaluate(csv_path, target_col, test_size, random_state, model_path)
    -> dict of evaluation metrics
load_pipeline(path) -> fitted sklearn Pipeline
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocess import load_and_preprocess

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "models/pipeline.pkl"

# Fixed production-grade XGBoost hyperparameters for tabular imbalanced data.
# scale_pos_weight is set dynamically at fit-time from the training label ratio.
XGB_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Return neg/pos ratio for XGBoost's scale_pos_weight."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        raise ValueError("[train] No positive samples found in y_train.")
    ratio = float(n_neg) / float(n_pos)
    print(f"[train] scale_pos_weight = {ratio:.3f}  (neg={n_neg}, pos={n_pos})")
    return ratio


def _build_pipeline(
    preprocessor,
    scale_pos_weight: float,
    random_state: int = 42,
) -> Pipeline:
    """Wrap preprocessor + XGBClassifier into a single sklearn Pipeline."""
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight, "random_state": random_state}
    clf = XGBClassifier(**params)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def _evaluate_on_test(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute held-out test metrics."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }


def _print_metrics(label: str, metrics: dict) -> None:
    print(f"\n[train] {label}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_and_evaluate(
    csv_path: str,
    target_col: str = "Is_AI_Assisted",
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: str = DEFAULT_MODEL_PATH,
) -> dict:
    """
    Full training pipeline: preprocess → CV evaluation → full fit → save artifact.

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV.
    target_col : str
        Name of the binary target column. Default: 'Is_AI_Assisted'.
    test_size : float
        Held-out test fraction. Default: 0.2.
    random_state : int
        Reproducibility seed. Default: 42.
    model_path : str
        Where to save the fitted Pipeline via joblib. Default: 'models/pipeline.pkl'.

    Returns
    -------
    dict
        Held-out test metrics: roc_auc, f1, precision, recall, accuracy.
    """
    # 1. Load and preprocess
    preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess(
        csv_path,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    # 2. Compute class-imbalance weight from training labels only
    spw = _compute_scale_pos_weight(y_train)

    # 3. Build the full pipeline
    pipeline = _build_pipeline(preprocessor, scale_pos_weight=spw, random_state=random_state)

    # 4. Stratified 5-fold cross-validation on training data
    #    (refit=False here — CV is for reporting only)
    print("[train] Running 5-fold stratified cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=["roc_auc", "f1_macro", "precision_macro", "recall_macro", "accuracy"],
        n_jobs=1,   # XGBoost already uses n_jobs=-1 internally
        return_train_score=False,
    )
    cv_metrics = {
        "roc_auc":   float(cv_results["test_roc_auc"].mean()),
        "f1":        float(cv_results["test_f1_macro"].mean()),
        "precision": float(cv_results["test_precision_macro"].mean()),
        "recall":    float(cv_results["test_recall_macro"].mean()),
        "accuracy":  float(cv_results["test_accuracy"].mean()),
    }
    _print_metrics("5-Fold CV (mean across folds)", cv_metrics)

    # 5. Final fit on full training set
    print("[train] Fitting final pipeline on full X_train ...")
    pipeline.fit(X_train, y_train)

    # 6. Held-out test evaluation
    test_metrics = _evaluate_on_test(pipeline, X_test, y_test)
    _print_metrics("Held-out Test Set", test_metrics)

    # 7. Save artifact
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"[train] Pipeline saved to: {model_path}")

    return test_metrics


def load_pipeline(path: str) -> Pipeline:
    """Load and return a previously saved sklearn Pipeline from disk."""
    pipeline = joblib.load(path)
    print(f"[train] Pipeline loaded from: {path}")
    return pipeline


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/ai_writing_detection_dataset.csv"
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL_PATH

    print(f"[train] Dataset : {csv_path}")
    print(f"[train] Output  : {model_path}")

    metrics = train_and_evaluate(csv_path, model_path=model_path)

    print("\n-- Final Test Metrics ----------------------------------------")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    pipeline = load_pipeline(model_path)
    print(f"\n[train] Pipeline steps: {[name for name, _ in pipeline.steps]}")
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -v
```
Expected: All tests pass (12 preprocess + 8 train = 20 total)

- [ ] **Step 3: Run the demo on the real dataset**

```bash
python -m src.train data/ai_writing_detection_dataset.csv models/pipeline.pkl
```
Expected: CV metrics printed, held-out test metrics printed, `Pipeline saved to: models/pipeline.pkl`

---

## Self-Review

| Spec requirement | Task |
|---|---|
| XGBoost classifier | Task 3 → `XGBClassifier` |
| `scale_pos_weight` from neg/pos ratio | Task 3 → `_compute_scale_pos_weight` |
| Fixed hyperparameters (no tuning) | Task 3 → `XGB_PARAMS` constant |
| Stratified 5-fold CV metrics | Task 3 → `cross_validate` with `StratifiedKFold` |
| ROC-AUC, F1, Precision, Recall, Accuracy | Task 3 → `_evaluate_on_test` + CV scoring |
| Held-out test evaluation | Task 3 → `_evaluate_on_test` |
| Save `models/pipeline.pkl` via joblib | Task 3 → `joblib.dump` |
| `load_pipeline(path)` | Task 3 → public function |
| `train_and_evaluate(...)` signature | Task 3 → public function |
| Preprocessor patch: `Is_AI_Assisted` default | Task 1 → `load_and_preprocess` `target_col` param |
| Preprocessor patch: extra log-transform cols | Task 1 → `_LOG_SKEWED_COLS` |
| Preprocessor patch: skip string mapping for already-binary target | Task 1 → `_encode_label` guard |
| Tests cover import, return types, keys, ranges, save/load, predict, predict_proba | Task 2 |

All requirements covered.
