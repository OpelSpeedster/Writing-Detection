"""
src/api.py
==========
FastAPI backend for the AI vs Human Academic Writing prediction service.

Startup
-------
The trained sklearn Pipeline is loaded once at startup from MODEL_PATH
(env var, default: models/pipeline.pkl) and stored on app.state.

Endpoints
---------
GET  /health          Liveness check
GET  /model/info      Pipeline metadata
POST /predict         Single-record prediction
POST /predict/batch   Batch predictions
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH: str = os.environ.get("MODEL_PATH", "models/pipeline.pkl")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

# All fields match the raw Kaggle dataset columns (Student_ID excluded — ID col).
# Numeric fields that have real NULLs in the dataset are Optional.
# Word_Count and Submission_Hour are always present (never null in dataset).


class PredictRequest(BaseModel):
    """Raw feature payload for a single writing sample."""

    # Categorical
    Submission_Type: Optional[str] = Field(
        None, description="Type of submission (Essay, Lab_Report, Research_Paper, Literature_Review)"
    )
    Academic_Level: Optional[str] = Field(
        None, description="Academic level (Undergraduate, Postgraduate, High_School)"
    )
    Primary_Language: Optional[str] = Field(
        None, description="Primary language (Native, Non_Native)"
    )
    Font_Family: Optional[str] = Field(
        None, description="Font used (Arial, Helvetica, Times New Roman, Calibri)"
    )

    # Required numerics
    Word_Count: int = Field(..., ge=0, description="Total word count")
    Submission_Hour: int = Field(..., ge=0, le=23, description="Hour of submission (0–23)")

    # Optional numerics (dataset has documented NULLs)
    Average_Sentence_Length: Optional[float] = Field(None, ge=0)
    Grammar_Errors: Optional[float] = Field(None, ge=0)
    Vocabulary_Richness: Optional[float] = Field(None, ge=0.0, le=1.0)
    Passive_Voice_Ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    Reading_Level: Optional[float] = Field(None, ge=0)
    Editing_Time: Optional[float] = Field(None, ge=0)
    Revision_Count: Optional[float] = Field(None, ge=0)
    Typing_Speed: Optional[float] = Field(None, ge=0)
    Plagiarism_Score: Optional[float] = Field(None, ge=0)
    Citation_Count: Optional[float] = Field(None, ge=0)
    Punctuation_Density: Optional[float] = Field(None, ge=0)
    Sentence_Complexity: Optional[float] = Field(None, ge=0)

    model_config = {"json_schema_extra": {
        "example": {
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
    }}


class PredictResponse(BaseModel):
    """Prediction result for a single writing sample."""
    label: str = Field(..., description='"AI" or "Human"')
    is_ai: int = Field(..., description="1 if AI-assisted, 0 if Human")
    probability_ai: float = Field(..., ge=0.0, le=1.0, description="Probability of AI authorship")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Max(p_ai, 1-p_ai) — model confidence")


class BatchPredictRequest(BaseModel):
    records: List[PredictRequest] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    count: int


# ---------------------------------------------------------------------------
# Lifespan — load pipeline at startup, release at shutdown
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pipeline loading — lifespan (local) + lazy (Vercel serverless via Mangum)
# Vercel serverless functions do not reliably run lifespan handlers, so every
# request path falls back to lazy disk load with a module-level cache.
# ---------------------------------------------------------------------------

_cached_pipeline = None


def _load_pipeline_from_disk():
    """Load pipeline from MODEL_PATH; return None instead of raising."""
    global _cached_pipeline
    if _cached_pipeline is not None:
        return _cached_pipeline
    try:
        _cached_pipeline = joblib.load(MODEL_PATH)
        print(f"[api] Pipeline lazily loaded from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[api] WARNING: pipeline not found at {MODEL_PATH}. /predict will return 503.")
        _cached_pipeline = None
    return _cached_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML pipeline on startup; unload on shutdown."""
    pipeline = None
    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"[api] Pipeline loaded from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[api] WARNING: pipeline not found at {MODEL_PATH}. /predict will return 503.")
    app.state.pipeline = pipeline
    global _cached_pipeline
    _cached_pipeline = pipeline
    yield
    app.state.pipeline = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI vs Human Writing Detector",
    description="Predict whether academic writing was AI-assisted or human-authored.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — required for Streamlit Cloud frontend calling the Vercel backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request_to_df(record: PredictRequest) -> pd.DataFrame:
    """Convert a PredictRequest to a single-row DataFrame with correct column names."""
    return pd.DataFrame([record.model_dump()])


def _make_prediction(pipeline, df: pd.DataFrame) -> PredictResponse:
    """Run the pipeline on a single-row DataFrame and return a PredictResponse."""
    is_ai: int = int(pipeline.predict(df)[0])
    proba: float = float(pipeline.predict_proba(df)[0, 1])
    confidence: float = float(max(proba, 1.0 - proba))
    label: str = "AI" if is_ai == 1 else "Human"
    return PredictResponse(
        label=label,
        is_ai=is_ai,
        probability_ai=round(proba, 6),
        confidence=round(confidence, 6),
    )


def _get_pipeline(request_state):
    """Return the loaded pipeline or raise 503 if not available.

    Checks lifespan state first, then falls back to lazy disk load so
    Vercel serverless (Mangum, no lifespan) still serves predictions.
    """
    pipeline = getattr(request_state, "pipeline", None)
    if pipeline is None:
        pipeline = _load_pipeline_from_disk()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model pipeline not loaded. Run src/train.py to generate models/pipeline.pkl.",
        )
    return pipeline


def _is_model_loaded(request_state) -> bool:
    """Non-raising check for /health (lifespan state or lazy disk load)."""
    if getattr(request_state, "pipeline", None) is not None:
        return True
    return _load_pipeline_from_disk() is not None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness check")
def health():
    """Returns 200 with model_loaded status. Always responds, even if model is missing."""
    return {"status": "ok", "model_loaded": _is_model_loaded(app.state)}


@app.get("/model/info", summary="Pipeline metadata")
def model_info():
    """Return the names of pipeline steps and input feature count."""
    pipeline = _get_pipeline(app.state)
    steps = [name for name, _ in pipeline.steps]

    # Dig into the preprocessor sub-pipeline to retrieve feature names seen at fit time
    preprocessor_pipe = pipeline.named_steps.get("preprocessor")
    n_features = None
    if preprocessor_pipe is not None and hasattr(preprocessor_pipe, "feature_names_in_"):
        n_features = int(preprocessor_pipe.feature_names_in_.shape[0])
    elif hasattr(pipeline, "feature_names_in_"):
        n_features = int(pipeline.feature_names_in_.shape[0])

    return {
        "pipeline_steps": steps,
        "n_features_in": n_features,
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=PredictResponse, summary="Single prediction")
def predict(request: PredictRequest):
    """
    Predict whether a single writing sample is AI-assisted or human-authored.

    Returns the predicted label, binary flag, AI probability, and model confidence.
    """
    pipeline = _get_pipeline(app.state)
    df = _request_to_df(request)
    return _make_prediction(pipeline, df)


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch predictions",
)
def predict_batch(request: BatchPredictRequest):
    """
    Predict for a batch of writing samples in a single call.

    The `records` list must contain at least one item (enforced by Pydantic min_length=1).
    """
    pipeline = _get_pipeline(app.state)
    df = pd.DataFrame([r.model_dump() for r in request.records])

    is_ai_arr = pipeline.predict(df).astype(int)
    proba_arr = pipeline.predict_proba(df)[:, 1]

    predictions = [
        PredictResponse(
            label="AI" if int(is_ai) == 1 else "Human",
            is_ai=int(is_ai),
            probability_ai=round(float(p), 6),
            confidence=round(float(max(p, 1.0 - p)), 6),
        )
        for is_ai, p in zip(is_ai_arr, proba_arr)
    ]

    return BatchPredictResponse(predictions=predictions, count=len(predictions))


# ---------------------------------------------------------------------------
# Entry point  (python -m src.api  or  uvicorn src.api:app --reload)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
