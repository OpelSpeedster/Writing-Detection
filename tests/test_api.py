"""
tests/test_api.py
=================
Tests for src/api.py — FastAPI prediction backend.

Uses FastAPI's TestClient (via httpx) for all requests.
The pipeline is loaded from models/pipeline.pkl which must already exist
(produced by running src/train.py on the real dataset).
"""
import os
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Minimal valid payload matching the real dataset schema
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "Submission_Type": "Essay",
    "Academic_Level": "Undergraduate",
    "Primary_Language": "Native",
    "Word_Count": 800,
    "Average_Sentence_Length": 18.0,
    "Grammar_Errors": 3.0,
    "Vocabulary_Richness": 0.6,
    "Passive_Voice_Ratio": 0.2,
    "Reading_Level": 12.0,
    "Editing_Time": 90.0,
    "Revision_Count": 5.0,
    "Typing_Speed": 45.0,
    "Plagiarism_Score": 1.5,
    "Citation_Count": 4.0,
    "Punctuation_Density": 0.07,
    "Sentence_Complexity": 2.0,
    "Font_Family": "Arial",
    "Submission_Hour": 14,
}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the real pipeline loaded."""
    from src.api import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_body(client):
    resp = client.get("/health")
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


# ---------------------------------------------------------------------------
# GET /model/info
# ---------------------------------------------------------------------------

def test_model_info_returns_200(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200


def test_model_info_keys(client):
    resp = client.get("/model/info")
    data = resp.json()
    assert "pipeline_steps" in data
    assert "n_features_in" in data
    assert isinstance(data["pipeline_steps"], list)


# ---------------------------------------------------------------------------
# POST /predict — happy path
# ---------------------------------------------------------------------------

def test_predict_returns_200(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200


def test_predict_response_keys(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    data = resp.json()
    assert "label" in data
    assert "is_ai" in data
    assert "probability_ai" in data
    assert "confidence" in data


def test_predict_label_is_valid_string(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    data = resp.json()
    assert data["label"] in ("AI", "Human")


def test_predict_is_ai_is_binary_int(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    data = resp.json()
    assert data["is_ai"] in (0, 1)


def test_predict_probability_in_range(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    data = resp.json()
    assert 0.0 <= data["probability_ai"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_with_null_optional_fields(client):
    """Nullable fields (dataset has real NULLs) should not crash the API."""
    payload = {**VALID_PAYLOAD, "Editing_Time": None, "Revision_Count": None}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_missing_required_field_returns_422(client):
    """Word_Count is required — omitting it must return 422 Unprocessable Entity."""
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "Word_Count"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------

def test_batch_predict_returns_200(client):
    resp = client.post("/predict/batch", json={"records": [VALID_PAYLOAD, VALID_PAYLOAD]})
    assert resp.status_code == 200


def test_batch_predict_response_keys(client):
    resp = client.post("/predict/batch", json={"records": [VALID_PAYLOAD]})
    data = resp.json()
    assert "predictions" in data
    assert "count" in data


def test_batch_predict_count_matches_input(client):
    n = 3
    resp = client.post("/predict/batch", json={"records": [VALID_PAYLOAD] * n})
    data = resp.json()
    assert data["count"] == n
    assert len(data["predictions"]) == n


def test_batch_predict_each_prediction_has_required_keys(client):
    resp = client.post("/predict/batch", json={"records": [VALID_PAYLOAD]})
    pred = resp.json()["predictions"][0]
    for key in ("label", "is_ai", "probability_ai", "confidence"):
        assert key in pred, f"Missing key: {key}"


def test_batch_predict_empty_list_is_rejected(client):
    """An empty batch is rejected by Pydantic min_length validation (422)."""
    resp = client.post("/predict/batch", json={"records": []})
    assert resp.status_code == 422
