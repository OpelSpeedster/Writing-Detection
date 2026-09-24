"""
src/train.py
============
Train an XGBoost binary classifier on the AI vs Human Academic Writing dataset.

Public API
----------
train_and_evaluate(csv_path, target_col, test_size, random_state, model_path)
    -> dict of held-out test metrics (roc_auc, f1, precision, recall, accuracy)
load_pipeline(path)
    -> fitted sklearn Pipeline (preprocessor + XGBClassifier)
"""
from __future__ import annotations

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

# Fixed production-ready XGBoost hyperparameters for tabular imbalanced data.
# scale_pos_weight is NOT here — it is computed dynamically from y_train.
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
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Return neg/pos count ratio for XGBoost's scale_pos_weight parameter."""
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError("[train] No positive samples in y_train — cannot compute scale_pos_weight.")
    ratio = float(n_neg) / float(n_pos)
    print(f"[train] scale_pos_weight = {ratio:.3f}  (neg={n_neg}, pos={n_pos})")
    return ratio


def _build_pipeline(
    preprocessor,
    scale_pos_weight: float,
    random_state: int = 42,
) -> Pipeline:
    """Combine fitted preprocessor + XGBClassifier into a single sklearn Pipeline."""
    params = {
        **XGB_PARAMS,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
    }
    clf = XGBClassifier(**params)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def _evaluate_on_test(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute and return held-out test metrics."""
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
    Full training pipeline: preprocess -> 5-fold CV report -> final fit -> save artifact.

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
        Destination for the serialized Pipeline. Default: 'models/pipeline.pkl'.

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

    # 2. Compute imbalance weight from training labels only (no leakage)
    spw = _compute_scale_pos_weight(y_train)

    # 3. Build pipeline
    pipeline = _build_pipeline(preprocessor, scale_pos_weight=spw, random_state=random_state)

    # 4. Stratified 5-fold CV on training data (for reporting — does not affect final model)
    print("[train] Running 5-fold stratified cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=["roc_auc", "f1_macro", "precision_macro", "recall_macro", "accuracy"],
        n_jobs=1,  # XGBoost already parallelises internally
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

    # 7. Persist artifact
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
# Demo / smoke-test  (python -m src.train [csv_path] [model_path])
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
