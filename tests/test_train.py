# tests/test_train.py
"""Tests for src/train.py -- XGBoost training pipeline."""
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


def make_sample_row() -> pd.DataFrame:
    """Single-row DataFrame for predict/predict_proba tests (no target column)."""
    return pd.DataFrame(
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
    preds = pipeline.predict(make_sample_row())
    assert preds.shape == (1,)
    assert preds[0] in (0, 1)


def test_load_pipeline_predict_proba(tmp_path):
    from src.train import train_and_evaluate, load_pipeline
    csv_path = make_train_csv(tmp_path, n=100)
    model_path = str(tmp_path / "pipeline.pkl")
    train_and_evaluate(csv_path, model_path=model_path)
    pipeline = load_pipeline(model_path)
    proba = pipeline.predict_proba(make_sample_row())
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-6
