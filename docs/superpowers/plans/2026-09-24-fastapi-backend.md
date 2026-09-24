# docs/superpowers/plans/2026-09-24-fastapi-backend.md

# FastAPI Backend — Implementation Plan

**Goal:** Build `src/api.py` — a FastAPI application that loads `models/pipeline.pkl`
at startup and exposes prediction, health, and model-info endpoints.

**Architecture:** Single module `src/api.py`. The pipeline is loaded once at startup
via FastAPI's `lifespan` context manager and stored as app state. All prediction
logic delegates to `pipeline.predict` / `pipeline.predict_proba` — no ML code
lives in the API layer. Input validation is done by Pydantic models.

**Tech Stack:** FastAPI, Uvicorn, Pydantic v2, httpx (test client), pytest

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | /health | Liveness check — returns `{ status: "ok" }` |
| GET  | /model/info | Model metadata: pipeline steps, feature names |
| POST | /predict | Single-record prediction |
| POST | /predict/batch | Batch predictions (list of records) |

## Request / Response schemas

**`PredictRequest`** — all fields from the raw dataset (target excluded):
```
Submission_Type, Academic_Level, Primary_Language, Word_Count,
Average_Sentence_Length, Grammar_Errors, Vocabulary_Richness,
Passive_Voice_Ratio, Reading_Level, Editing_Time, Revision_Count,
Typing_Speed, Plagiarism_Score, Citation_Count, Punctuation_Density,
Sentence_Complexity, Font_Family, Submission_Hour
```

**`PredictResponse`**:
```json
{ "label": "AI" | "Human", "is_ai": 0|1, "probability_ai": 0.0–1.0, "confidence": 0.0–1.0 }
```

**`BatchPredictRequest`**: `{ "records": [PredictRequest, ...] }`
**`BatchPredictResponse`**: `{ "predictions": [PredictResponse, ...], "count": int }`

## Global Constraints
- Model loaded from `MODEL_PATH` env var (default: `models/pipeline.pkl`)
- 503 returned if pipeline not loaded when a predict request arrives
- All numeric fields have sensible `ge`/`le` validators
- Categorical fields are `Optional[str]` (dataset has real NULLs)
