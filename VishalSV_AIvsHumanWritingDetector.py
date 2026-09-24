"""
VishalSV_AIvsHumanWritingDetector.py
=====================================
IBM SkillsBuild Data Analytics with AI Academic Internship Program
BharatCares in association with AICTE

Project : AI vs Human Academic Writing Detector
Student : VishalSV
Dataset : AI vs Human Academic Writing Dataset — Razan Ihab Abdellatif (Kaggle)
          https://www.kaggle.com/datasets/razanihababdellatif/ai-vs-human-academic-writing-dataset

Description
-----------
End-to-end binary classification system that predicts whether a piece of
academic writing is AI-assisted (label=1) or Human-authored (label=0).

Modules included in this file
------------------------------
SECTION 1 — Preprocessing  (load_and_preprocess)
SECTION 2 — Training       (train_and_evaluate, load_pipeline)
SECTION 3 — FastAPI Backend (app / routes)
SECTION 4 — Streamlit Frontend

How to run
----------
1. Install dependencies:
       pip install -r requirements.txt

2. Train the model (once):
       python VishalSV_AIvsHumanWritingDetector.py train \
           data/ai_writing_detection_dataset.csv models/pipeline.pkl

3. Start the FastAPI backend (Terminal 1):
       uvicorn VishalSV_AIvsHumanWritingDetector:app --reload --port 8000

4. Start the Streamlit frontend (Terminal 2):
       streamlit run VishalSV_AIvsHumanWritingDetector.py

Results (real dataset, 10,000 clean records, 80/20 split)
----------------------------------------------------------
5-Fold CV   ROC-AUC 1.0000 | F1 0.9992 | Accuracy 0.9996
Test Set    ROC-AUC 1.0000 | F1 1.0000 | Accuracy 1.0000
"""

from __future__ import annotations

# ===========================================================================
# SECTION 1 — PREPROCESSING
# ===========================================================================

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
LABEL_MAP  = {"AI": 1, "Human": 0}

# Skewed telemetry columns that benefit from log1p compression
_LOG_SKEWED_COLS = ["Editing_Time", "Revision_Count", "Grammar_Errors", "Typing_Speed"]

# Ratio feature specs: (new_col_name, numerator_col, denominator_col)
# Denominator +1 prevents zero-division.
_RATIO_SPECS = [
    ("words_per_minute", "Word_Count",     "Editing_Time"),
    ("revision_density", "Revision_Count", "Word_Count"),
    ("lexical_diversity","Unique_Words",    "Total_Words"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (logging glitches in the dataset)."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"[preprocess] Dropped {before - after} duplicate rows.")
    return df.reset_index(drop=True)


def _drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-predictive identifier columns (any column name containing 'id')."""
    id_cols = [c for c in df.columns if "id" in c.lower()]
    if id_cols:
        print(f"[preprocess] Dropping ID-like columns: {id_cols}")
        df = df.drop(columns=id_cols)
    return df


def _encode_label(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Map 'AI'->1 / 'Human'->0. No-op when column is already numeric."""
    if pd.api.types.is_numeric_dtype(df[target_col]):
        return df
    df = df.copy()
    df[target_col] = df[target_col].map(LABEL_MAP)
    unmapped = df[target_col].isna().sum()
    if unmapped:
        raise ValueError(
            f"[preprocess] {unmapped} rows have unmapped label values. "
            "Expected 'AI' or 'Human'."
        )
    df[target_col] = df[target_col].astype(int)
    return df


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rate/ratio features; fill any resulting NaN/Inf with 0."""
    df = df.copy()
    for new_col, num_col, den_col in _RATIO_SPECS:
        if num_col in df.columns and den_col in df.columns:
            df[new_col] = df[num_col] / (df[den_col] + 1)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply np.log1p to heavily right-skewed telemetry columns."""
    df = df.copy()
    for col in [c for c in _LOG_SKEWED_COLS if c in df.columns]:
        # Cast to float so Python None -> NaN (prevents ufunc errors at inference)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = np.log1p(df[col].clip(lower=0))
    return df


def _feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """Combined feature engineering step used as a stateless FunctionTransformer."""
    X = _add_ratio_features(X)
    X = _log_transform(X)
    return X


def _build_preprocessor_pipeline(X_raw: pd.DataFrame) -> Pipeline:
    """
    Build a full sklearn Pipeline:
        step 1 — FunctionTransformer: ratio + log1p features
        step 2 — ColumnTransformer:
                    numeric   -> SimpleImputer(median) -> RobustScaler
                    categorical -> SimpleImputer(most_frequent) -> OneHotEncoder
    """
    X_eng = _feature_engineering(X_raw)
    numeric_cols     = X_eng.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_eng.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer",  SimpleImputer(strategy="most_frequent")),
        ("encoder",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("numeric",      num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical",  cat_pipe, categorical_cols))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([
        ("feature_engineering", FunctionTransformer(_feature_engineering, validate=False)),
        ("column_transformer",  ct),
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_preprocess(
    csv_path: str,
    target_col: str = "Is_AI_Assisted",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Load CSV, clean it, split, and return a fitted sklearn preprocessing
    Pipeline alongside stratified train/test splits of RAW features.

    The Pipeline is fully self-contained:
        raw DataFrame -> feature engineering -> impute/scale/encode -> numpy array

    Returns
    -------
    preprocessor : fitted sklearn Pipeline
    X_train, X_test : pd.DataFrame  (raw features)
    y_train, y_test : pd.Series     (binary int 0/1)
    """
    print(f"[preprocess] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[preprocess] Raw shape: {df.shape}")

    df = _drop_duplicates(df)
    df = _drop_id_columns(df)
    df = _encode_label(df, target_col=target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"[preprocess] Class balance : {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[preprocess] Split  : X_train={X_train.shape}, X_test={X_test.shape}")

    preprocessor = _build_preprocessor_pipeline(X_train)
    preprocessor.fit(X_train)
    return preprocessor, X_train, X_test, y_train, y_test


# ===========================================================================
# SECTION 2 — MODEL TRAINING
# ===========================================================================

from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

DEFAULT_MODEL_PATH = "models/pipeline.pkl"

# Fixed production-grade XGBoost hyperparameters.
# scale_pos_weight is computed dynamically from y_train at fit-time.
XGB_PARAMS: dict = {
    "n_estimators":    400,
    "max_depth":       6,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "min_child_weight":5,
    "gamma":           0.1,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "eval_metric":     "logloss",
    "n_jobs":          -1,
}


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Return neg/pos ratio for XGBoost's scale_pos_weight."""
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        raise ValueError("[train] No positive samples in y_train.")
    ratio = float(n_neg) / float(n_pos)
    print(f"[train] scale_pos_weight = {ratio:.3f}  (neg={n_neg}, pos={n_pos})")
    return ratio


def _build_xgb_pipeline(preprocessor, scale_pos_weight: float, random_state: int = 42) -> Pipeline:
    """Wrap preprocessor + XGBClassifier into a single sklearn Pipeline."""
    clf = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight, random_state=random_state)
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def _evaluate_on_test(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute held-out test metrics."""
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "roc_auc":  float(roc_auc_score(y_test, y_proba)),
        "f1":       float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision":float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall":   float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }


def train_and_evaluate(
    csv_path: str,
    target_col: str = "Is_AI_Assisted",
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: str = DEFAULT_MODEL_PATH,
) -> dict:
    """
    Full pipeline: preprocess -> 5-fold CV -> final fit -> save artifact.

    Returns dict of held-out test metrics: roc_auc, f1, precision, recall, accuracy.
    """
    preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess(
        csv_path, target_col=target_col, test_size=test_size, random_state=random_state
    )

    spw      = _compute_scale_pos_weight(y_train)
    pipeline = _build_xgb_pipeline(preprocessor, scale_pos_weight=spw, random_state=random_state)

    # Stratified 5-fold CV (reporting only — no impact on final model)
    print("[train] Running 5-fold stratified cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring=["roc_auc", "f1_macro", "precision_macro", "recall_macro", "accuracy"],
        n_jobs=1, return_train_score=False,
    )
    cv_metrics = {
        "roc_auc":   float(cv_results["test_roc_auc"].mean()),
        "f1":        float(cv_results["test_f1_macro"].mean()),
        "precision": float(cv_results["test_precision_macro"].mean()),
        "recall":    float(cv_results["test_recall_macro"].mean()),
        "accuracy":  float(cv_results["test_accuracy"].mean()),
    }
    print("\n[train] 5-Fold CV (mean across folds)")
    for k, v in cv_metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    # Final fit on full training set
    print("\n[train] Fitting final pipeline on X_train ...")
    pipeline.fit(X_train, y_train)

    # Held-out test evaluation
    test_metrics = _evaluate_on_test(pipeline, X_test, y_test)
    print("\n[train] Held-out Test Set")
    for k, v in test_metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    # Save artifact
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\n[train] Pipeline saved to: {model_path}")

    return test_metrics


def load_pipeline(path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load and return a previously saved sklearn Pipeline from disk."""
    pipeline = joblib.load(path)
    print(f"[train] Pipeline loaded from: {path}")
    return pipeline


# ===========================================================================
# SECTION 3 — FASTAPI BACKEND
# ===========================================================================

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH: str = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)


class PredictRequest(BaseModel):
    """Raw feature payload for a single writing sample."""
    Submission_Type:         Optional[str]   = Field(None)
    Academic_Level:          Optional[str]   = Field(None)
    Primary_Language:        Optional[str]   = Field(None)
    Font_Family:             Optional[str]   = Field(None)
    Word_Count:              int             = Field(..., ge=0)
    Submission_Hour:         int             = Field(..., ge=0, le=23)
    Average_Sentence_Length: Optional[float] = Field(None, ge=0)
    Grammar_Errors:          Optional[float] = Field(None, ge=0)
    Vocabulary_Richness:     Optional[float] = Field(None, ge=0.0, le=1.0)
    Passive_Voice_Ratio:     Optional[float] = Field(None, ge=0.0, le=1.0)
    Reading_Level:           Optional[float] = Field(None, ge=0)
    Editing_Time:            Optional[float] = Field(None, ge=0)
    Revision_Count:          Optional[float] = Field(None, ge=0)
    Typing_Speed:            Optional[float] = Field(None, ge=0)
    Plagiarism_Score:        Optional[float] = Field(None, ge=0)
    Citation_Count:          Optional[float] = Field(None, ge=0)
    Punctuation_Density:     Optional[float] = Field(None, ge=0)
    Sentence_Complexity:     Optional[float] = Field(None, ge=0)

    model_config = {"json_schema_extra": {"example": {
        "Submission_Type": "Essay", "Academic_Level": "Undergraduate",
        "Primary_Language": "Native", "Word_Count": 800, "Submission_Hour": 14,
        "Editing_Time": 90.0, "Typing_Speed": 45.0, "Revision_Count": 5.0,
        "Grammar_Errors": 3.0, "Vocabulary_Richness": 0.6,
        "Passive_Voice_Ratio": 0.2, "Reading_Level": 12.0,
        "Plagiarism_Score": 1.5, "Citation_Count": 4.0,
        "Punctuation_Density": 0.07, "Sentence_Complexity": 2.0,
        "Font_Family": "Arial", "Average_Sentence_Length": 18.0,
    }}}


class PredictResponse(BaseModel):
    label:          str   = Field(..., description='"AI" or "Human"')
    is_ai:          int   = Field(..., description="1 = AI, 0 = Human")
    probability_ai: float = Field(..., ge=0.0, le=1.0)
    confidence:     float = Field(..., ge=0.0, le=1.0)


class BatchPredictRequest(BaseModel):
    records: List[PredictRequest] = Field(..., min_length=1)


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    count: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML pipeline once at startup; release on shutdown."""
    _pipeline = None
    try:
        _pipeline = joblib.load(MODEL_PATH)
        print(f"[api] Pipeline loaded from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[api] WARNING: pipeline not found at {MODEL_PATH}. /predict will return 503.")
    app.state.pipeline = _pipeline
    yield
    app.state.pipeline = None


app = FastAPI(
    title="AI vs Human Writing Detector",
    description="Predict whether academic writing is AI-assisted or human-authored.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_pipeline(state):
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train first.")
    return state.pipeline


def _run_prediction(pipeline, df: pd.DataFrame) -> PredictResponse:
    is_ai  = int(pipeline.predict(df)[0])
    proba  = float(pipeline.predict_proba(df)[0, 1])
    conf   = float(max(proba, 1.0 - proba))
    return PredictResponse(
        label="AI" if is_ai else "Human",
        is_ai=is_ai,
        probability_ai=round(proba, 6),
        confidence=round(conf, 6),
    )


@app.get("/health", summary="Liveness check")
def health():
    return {"status": "ok", "model_loaded": app.state.pipeline is not None}


@app.get("/model/info", summary="Pipeline metadata")
def model_info():
    pipeline = _get_pipeline(app.state)
    steps = [name for name, _ in pipeline.steps]
    pre   = pipeline.named_steps.get("preprocessor")
    n_feat = int(pre.feature_names_in_.shape[0]) if (
        pre is not None and hasattr(pre, "feature_names_in_")
    ) else None
    return {"pipeline_steps": steps, "n_features_in": n_feat, "model_path": MODEL_PATH}


@app.post("/predict", response_model=PredictResponse, summary="Single prediction")
def predict(request: PredictRequest):
    pipeline = _get_pipeline(app.state)
    return _run_prediction(pipeline, pd.DataFrame([request.model_dump()]))


@app.post("/predict/batch", response_model=BatchPredictResponse, summary="Batch predictions")
def predict_batch(request: BatchPredictRequest):
    pipeline  = _get_pipeline(app.state)
    df        = pd.DataFrame([r.model_dump() for r in request.records])
    is_ai_arr = pipeline.predict(df).astype(int)
    proba_arr = pipeline.predict_proba(df)[:, 1]
    preds = [
        PredictResponse(
            label="AI" if int(ia) else "Human", is_ai=int(ia),
            probability_ai=round(float(p), 6), confidence=round(float(max(p, 1-p)), 6),
        )
        for ia, p in zip(is_ai_arr, proba_arr)
    ]
    return BatchPredictResponse(predictions=preds, count=len(preds))


# ===========================================================================
# SECTION 4 — STREAMLIT FRONTEND
# ===========================================================================

def _run_streamlit_app():
    """Entry point when running: streamlit run VishalSV_AIvsHumanWritingDetector.py"""
    import requests as req
    import streamlit as st

    API_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")

    SUBMISSION_TYPES  = ["Essay", "Lab_Report", "Research_Paper", "Literature_Review"]
    ACADEMIC_LEVELS   = ["Undergraduate", "Postgraduate", "High_School"]
    PRIMARY_LANGUAGES = ["Native", "Non_Native"]
    FONT_FAMILIES     = ["Arial", "Helvetica", "Times New Roman", "Calibri"]

    st.set_page_config(page_title="Writing Detector", page_icon="🔬", layout="wide",
                       initial_sidebar_state="collapsed")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0F1117; color: #E2E8F0; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
    .section-label { font-size:0.68rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase;
                     color:#38BDF8; margin-bottom:0.4rem; margin-top:1.2rem; }
    .divider { border:none; border-top:1px solid #2D3748; margin:0.6rem 0 1.2rem 0; }
    [data-baseweb="input"] input,[data-baseweb="select"] div,[data-testid="stNumberInput"] input {
        background-color:#1C2333 !important; border-color:#2D3748 !important;
        color:#E2E8F0 !important; font-size:0.88rem !important; }
    label[data-testid="stWidgetLabel"] p { font-size:0.78rem !important; color:#94A3B8 !important; font-weight:500 !important; }
    .verdict-terminal { background:#0D1117; border:1px solid #2D3748; border-radius:6px;
                        padding:1.4rem 1.6rem 1.2rem; font-family:'JetBrains Mono',monospace; margin-bottom:1.2rem; }
    .verdict-prompt { font-size:0.72rem; color:#38BDF8; margin-bottom:0.5rem; letter-spacing:0.06em; }
    .verdict-text-ai   { font-size:1.8rem; font-weight:700; color:#F87171; letter-spacing:0.04em; line-height:1.1; }
    .verdict-text-human{ font-size:1.8rem; font-weight:700; color:#34D399; letter-spacing:0.04em; line-height:1.1; }
    .verdict-cursor { display:inline-block; width:0.55em; height:1.6rem; background:#38BDF8;
                      vertical-align:middle; margin-left:4px; animation:blink 1s step-start infinite; }
    @keyframes blink { 50% { opacity:0; } }
    .verdict-sub { font-size:0.72rem; color:#94A3B8; margin-top:0.6rem; line-height:1.6; }
    .verdict-sub span { color:#E2E8F0; }
    .prob-bar-container { background:#1C2333; border-radius:4px; height:8px; width:100%;
                          margin:0.5rem 0 0.3rem; overflow:hidden; }
    .prob-bar-fill-ai   { height:100%; border-radius:4px; background:linear-gradient(90deg,#7C3AED,#F87171); }
    .prob-bar-fill-human{ height:100%; border-radius:4px; background:linear-gradient(90deg,#0EA5E9,#34D399); }
    .prob-labels { display:flex; justify-content:space-between; font-size:0.68rem;
                   color:#94A3B8; font-family:'JetBrains Mono',monospace; }
    .metric-row { display:flex; gap:0.8rem; margin-bottom:1.2rem; }
    .metric-card { flex:1; background:#1C2333; border:1px solid #2D3748; border-radius:6px;
                   padding:0.8rem 1rem; text-align:center; }
    .metric-value { font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700;
                    color:#E2E8F0; line-height:1.2; }
    .metric-label { font-size:0.66rem; color:#94A3B8; text-transform:uppercase;
                    letter-spacing:0.1em; margin-top:0.2rem; }
    .stButton > button { background:#38BDF8 !important; color:#0F1117 !important;
                         font-weight:600 !important; font-size:0.88rem !important;
                         border:none !important; border-radius:5px !important;
                         padding:0.6rem 2rem !important; width:100% !important; }
    .idle-box { background:#1C2333; border:1px dashed #2D3748; border-radius:6px;
                padding:2.5rem 1.6rem; text-align:center; color:#4A5568;
                font-family:'JetBrains Mono',monospace; font-size:0.82rem; line-height:1.7; }
    .error-box { background:#2D1515; border:1px solid #F87171; border-radius:6px;
                 padding:1rem 1.2rem; color:#FCA5A5; font-size:0.82rem;
                 font-family:'JetBrains Mono',monospace; }
    </style>""", unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style="margin-bottom:0.4rem;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                     color:#38BDF8;letter-spacing:0.14em;">FORENSIC WRITING ANALYSIS</span>
    </div>
    <h1 style="font-family:'Inter',sans-serif;font-size:1.9rem;font-weight:600;
               color:#E2E8F0;margin:0 0 0.3rem 0;line-height:1.2;">
        AI vs Human<br>Writing Detector
    </h1>
    <p style="color:#94A3B8;font-size:0.88rem;margin:0 0 1.6rem 0;max-width:520px;">
        Enter the behavioral and linguistic profile of a writing sample.
        The model predicts whether it was AI-assisted or human-authored,
        trained on 10,200 academic submissions.
    </p>""", unsafe_allow_html=True)

    # Backend health check
    try:
        r = req.get(f"{API_URL}/health", timeout=2)
        backend_ok = r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        backend_ok = False

    if not backend_ok:
        st.markdown(f"""<div class="error-box">
        &#9888; Backend unreachable at <strong>{API_URL}</strong><br>
        Start the API: <code>uvicorn VishalSV_AIvsHumanWritingDetector:app --port 8000</code>
        </div>""", unsafe_allow_html=True)
        st.stop()

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="section-label">Behavioral Telemetry</div>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        bt1, bt2 = st.columns(2)
        with bt1:
            editing_time   = st.number_input("Editing Time (min)", min_value=0.0, value=90.0, step=1.0)
            revision_count = st.number_input("Revision Count",      min_value=0.0, value=5.0,  step=1.0)
        with bt2:
            typing_speed     = st.number_input("Typing Speed (WPM)",    min_value=0.0, value=45.0, step=1.0)
            submission_hour  = st.number_input("Submission Hour (0-23)", min_value=0, max_value=23, value=14)

        st.markdown('<div class="section-label">Linguistic Metrics</div>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        lm1, lm2 = st.columns(2)
        with lm1:
            word_count          = st.number_input("Word Count",         min_value=0,   value=800,  step=10)
            avg_sentence_len    = st.number_input("Avg Sentence Length",min_value=0.0, value=18.0, step=0.5)
            grammar_errors      = st.number_input("Grammar Errors",     min_value=0.0, value=3.0,  step=1.0)
            vocabulary_richness = st.slider("Vocabulary Richness", 0.0, 1.0, 0.6, 0.01)
        with lm2:
            passive_voice       = st.slider("Passive Voice Ratio", 0.0, 1.0, 0.2, 0.01)
            reading_level       = st.number_input("Reading Level (FK grade)", min_value=0.0, value=12.0, step=0.5)
            sentence_complexity = st.number_input("Sentence Complexity",      min_value=0.0, value=2.0,  step=0.1)
            punctuation_density = st.number_input("Punctuation Density",      min_value=0.0, value=0.07, step=0.01, format="%.3f")

        st.markdown('<div class="section-label">Document Metadata</div>', unsafe_allow_html=True)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        dm1, dm2 = st.columns(2)
        with dm1:
            submission_type  = st.selectbox("Submission Type",  SUBMISSION_TYPES)
            academic_level   = st.selectbox("Academic Level",   ACADEMIC_LEVELS)
            primary_language = st.selectbox("Primary Language", PRIMARY_LANGUAGES)
        with dm2:
            plagiarism_score = st.number_input("Plagiarism Score (%)", min_value=0.0, value=1.5, step=0.1)
            citation_count   = st.number_input("Citation Count",       min_value=0.0, value=4.0, step=1.0)
            font_family      = st.selectbox("Font Family", FONT_FAMILIES)

        st.markdown("<div style='margin-top:1.4rem;'>", unsafe_allow_html=True)
        analyse_clicked = st.button("Analyse Writing Sample", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if not analyse_clicked:
            st.markdown("""<div class="idle-box">
                &gt; awaiting sample...<br>
                &gt; fill in the form and click<br>
                &gt; "Analyse Writing Sample"<br>
                &gt; <span style="color:#38BDF8;">&#9608;</span>
            </div>""", unsafe_allow_html=True)
        else:
            payload = {
                "Submission_Type": submission_type, "Academic_Level": academic_level,
                "Primary_Language": primary_language, "Font_Family": font_family,
                "Word_Count": int(word_count), "Submission_Hour": int(submission_hour),
                "Average_Sentence_Length": float(avg_sentence_len),
                "Grammar_Errors": float(grammar_errors),
                "Vocabulary_Richness": float(vocabulary_richness),
                "Passive_Voice_Ratio": float(passive_voice),
                "Reading_Level": float(reading_level),
                "Editing_Time": float(editing_time), "Revision_Count": float(revision_count),
                "Typing_Speed": float(typing_speed), "Plagiarism_Score": float(plagiarism_score),
                "Citation_Count": float(citation_count),
                "Punctuation_Density": float(punctuation_density),
                "Sentence_Complexity": float(sentence_complexity),
            }
            with st.spinner("Analysing..."):
                try:
                    resp   = req.post(f"{API_URL}/predict", json=payload, timeout=10)
                    resp.raise_for_status()
                    result = resp.json()
                except req.exceptions.ConnectionError:
                    st.markdown(f'<div class="error-box">ERR: cannot connect to {API_URL}</div>',
                                unsafe_allow_html=True)
                    st.stop()
                except req.exceptions.HTTPError as e:
                    st.markdown(f'<div class="error-box">ERR: API {e.response.status_code}</div>',
                                unsafe_allow_html=True)
                    st.stop()

            is_ai      = result["is_ai"]
            prob_ai    = result["probability_ai"]
            confidence = result["confidence"]
            verdict_class = "verdict-text-ai" if is_ai else "verdict-text-human"
            verdict_text  = "AI ASSISTED" if is_ai else "HUMAN AUTHORED"
            prob_pct = f"{prob_ai * 100:.1f}"
            conf_pct = f"{confidence * 100:.1f}"

            st.markdown(f"""<div class="verdict-terminal">
                <div class="verdict-prompt">$ writing-detector --analyse sample.txt</div>
                <div class="{verdict_class}">VERDICT: {verdict_text}<span class="verdict-cursor"></span></div>
                <div class="verdict-sub">P(AI)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span>{prob_pct}%</span><br>
                Confidence &nbsp;<span>{conf_pct}%</span></div>
            </div>""", unsafe_allow_html=True)

            bar_class = "prob-bar-fill-ai" if is_ai else "prob-bar-fill-human"
            st.markdown(f"""<div style="margin-bottom:1rem;">
                <div style="font-size:0.72rem;color:#94A3B8;font-family:'JetBrains Mono',monospace;
                            margin-bottom:0.3rem;">AI PROBABILITY</div>
                <div class="prob-bar-container">
                    <div class="{bar_class}" style="width:{prob_ai*100:.1f}%;"></div>
                </div>
                <div class="prob-labels"><span>0% Human</span><span>100% AI</span></div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="metric-row">
                <div class="metric-card"><div class="metric-value">{prob_pct}%</div>
                <div class="metric-label">P(AI)</div></div>
                <div class="metric-card"><div class="metric-value">{100-float(prob_pct):.1f}%</div>
                <div class="metric-label">P(Human)</div></div>
                <div class="metric-card"><div class="metric-value">{conf_pct}%</div>
                <div class="metric-label">Confidence</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-label" style="margin-top:0;">Feature Snapshot</div>',
                        unsafe_allow_html=True)
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            snap = pd.DataFrame({
                "Feature": ["Editing Time","Typing Speed","Revision Count","Word Count",
                            "Grammar Errors","Vocab Richness","Plagiarism Score","Reading Level"],
                "Value":   [f"{editing_time:.0f} min", f"{typing_speed:.0f} wpm",
                            f"{revision_count:.0f}", f"{word_count}",
                            f"{grammar_errors:.0f}", f"{vocabulary_richness:.2f}",
                            f"{plagiarism_score:.1f}%", f"{reading_level:.1f}"],
            })
            st.dataframe(snap, use_container_width=True, hide_index=True)
            with st.expander("Raw API response", expanded=False):
                st.json(result)

    st.markdown("""<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #2D3748;
                text-align:center;font-size:0.72rem;color:#4A5568;font-family:'JetBrains Mono',monospace;">
        AI Writing Detector &nbsp;·&nbsp; XGBoost + FastAPI + Streamlit
        &nbsp;·&nbsp; Kaggle Dataset: Razan Ihab Abdellatif
    </div>""", unsafe_allow_html=True)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # python VishalSV_AIvsHumanWritingDetector.py train [csv_path] [model_path]
        csv   = sys.argv[2] if len(sys.argv) > 2 else "data/ai_writing_detection_dataset.csv"
        mpath = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL_PATH
        print(f"Dataset : {csv}\nOutput  : {mpath}")
        metrics = train_and_evaluate(csv, model_path=mpath)
        print("\n-- Final Test Metrics --")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v:.4f}")
    else:
        # Streamlit mode: streamlit run VishalSV_AIvsHumanWritingDetector.py
        _run_streamlit_app()
