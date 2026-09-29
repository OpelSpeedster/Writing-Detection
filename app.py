"""
app.py
======
Streamlit frontend for the AI vs Human Academic Writing Detector.

Two inference modes (automatic):
1. Backend mode — POSTs to the FastAPI backend (src/api.py) at API_BASE_URL,
   when that server is reachable and has the model loaded.
2. Standalone mode (Streamlit Cloud) — loads models/pipeline.pkl directly
   with joblib, no backend server required. This is the permanent-deploy path.

Env vars:
  API_BASE_URL — backend URL (default: http://localhost:8000).
                 Set to "" to force standalone mode.
  MODEL_PATH   — pipeline path for standalone mode (default: models/pipeline.pkl).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _resolve_setting(name: str, default: str) -> str:
    """Secrets (.streamlit/secrets.toml on Cloud) take precedence over env vars."""
    try:
        val = st.secrets.get(name, None)
        if val is not None and str(val).strip() != "":
            return str(val).strip()
        if val is not None:  # explicit empty string forces standalone mode
            return ""
    except Exception:
        pass
    return os.environ.get(name, default)


API_BASE_URL: str = _resolve_setting("API_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL_PATH: str = _resolve_setting("MODEL_PATH", "models/pipeline.pkl")

USE_BACKEND: bool = bool(API_BASE_URL)  # empty string forces standalone mode

SUBMISSION_TYPES   = ["Essay", "Lab_Report", "Research_Paper", "Literature_Review"]
ACADEMIC_LEVELS    = ["Undergraduate", "Postgraduate", "High_School"]
PRIMARY_LANGUAGES  = ["Native", "Non_Native"]
FONT_FAMILIES      = ["Arial", "Helvetica", "Times New Roman", "Calibri"]

# ---------------------------------------------------------------------------
# Page config & global CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Writing Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* ── Google Fonts ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;700&display=swap');

    /* ── Global reset ──────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0F1117;
        color: #E2E8F0;
    }

    /* ── Hide Streamlit chrome ─────────────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

    /* ── Section headings ──────────────────────────────────────── */
    .section-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #38BDF8;
        margin-bottom: 0.4rem;
        margin-top: 1.2rem;
    }
    .divider {
        border: none;
        border-top: 1px solid #2D3748;
        margin: 0.6rem 0 1.2rem 0;
    }

    /* ── Input overrides ───────────────────────────────────────── */
    [data-baseweb="input"] input,
    [data-baseweb="select"] div,
    [data-testid="stNumberInput"] input {
        background-color: #1C2333 !important;
        border-color: #2D3748 !important;
        color: #E2E8F0 !important;
        font-size: 0.88rem !important;
    }
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.78rem !important;
        color: #94A3B8 !important;
        font-weight: 500 !important;
    }

    /* ── Verdict terminal ──────────────────────────────────────── */
    .verdict-terminal {
        background: #0D1117;
        border: 1px solid #2D3748;
        border-radius: 6px;
        padding: 1.4rem 1.6rem 1.2rem;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 1.2rem;
    }
    .verdict-prompt {
        font-size: 0.72rem;
        color: #38BDF8;
        margin-bottom: 0.5rem;
        letter-spacing: 0.06em;
    }
    .verdict-text-ai {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F87171;
        letter-spacing: 0.04em;
        line-height: 1.1;
    }
    .verdict-text-human {
        font-size: 1.8rem;
        font-weight: 700;
        color: #34D399;
        letter-spacing: 0.04em;
        line-height: 1.1;
    }
    .verdict-cursor {
        display: inline-block;
        width: 0.55em;
        height: 1.6rem;
        background: #38BDF8;
        vertical-align: middle;
        margin-left: 4px;
        animation: blink 1s step-start infinite;
    }
    @keyframes blink { 50% { opacity: 0; } }
    .verdict-sub {
        font-size: 0.72rem;
        color: #94A3B8;
        margin-top: 0.6rem;
        line-height: 1.6;
    }
    .verdict-sub span { color: #E2E8F0; }

    /* ── Probability bar ───────────────────────────────────────── */
    .prob-bar-container {
        background: #1C2333;
        border-radius: 4px;
        height: 8px;
        width: 100%;
        margin: 0.5rem 0 0.3rem;
        overflow: hidden;
    }
    .prob-bar-fill-ai {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #7C3AED, #F87171);
        transition: width 0.5s ease;
    }
    .prob-bar-fill-human {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #0EA5E9, #34D399);
        transition: width 0.5s ease;
    }
    .prob-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.68rem;
        color: #94A3B8;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Metric cards ──────────────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        flex: 1;
        background: #1C2333;
        border: 1px solid #2D3748;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: #E2E8F0;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.66rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.2rem;
    }

    /* ── Submit button ─────────────────────────────────────────── */
    .stButton > button {
        background: #38BDF8 !important;
        color: #0F1117 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        letter-spacing: 0.04em;
        transition: opacity 0.15s;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* ── Idle placeholder ──────────────────────────────────────── */
    .idle-box {
        background: #1C2333;
        border: 1px dashed #2D3748;
        border-radius: 6px;
        padding: 2.5rem 1.6rem;
        text-align: center;
        color: #4A5568;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        line-height: 1.7;
    }

    /* ── Error box ─────────────────────────────────────────────── */
    .error-box {
        background: #2D1515;
        border: 1px solid #F87171;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        color: #FCA5A5;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Inference helpers — backend API with standalone fallback
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _check_backend() -> bool:
    """Return True if the backend /health endpoint is reachable with model loaded."""
    if not USE_BACKEND:
        return False
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        return r.status_code == 200 and r.json().get("model_loaded", False)
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def _load_local_pipeline():
    """Load the sklearn Pipeline from MODEL_PATH once; return None if missing."""
    try:
        import joblib

        path = Path(MODEL_PATH)
        if not path.is_file():
            # Fallback: resolve relative to this file (Streamlit Cloud cwd can vary)
            path = Path(__file__).parent / MODEL_PATH
        if not path.is_file():
            return None
        return joblib.load(path)
    except Exception:
        return None


def _predict_via_api(payload: dict) -> dict:
    """POST to /predict and return the response dict, or raise on error."""
    r = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def _predict_local(payload: dict) -> dict:
    """Run the local pipeline on a single payload; mirrors src/api.py logic."""
    import pandas as pd

    pipeline = _load_local_pipeline()
    if pipeline is None:
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run src/train.py first.")
    df = pd.DataFrame([payload])
    is_ai = int(pipeline.predict(df)[0])
    proba = float(pipeline.predict_proba(df)[0, 1])
    conf = float(max(proba, 1.0 - proba))
    return {
        "label": "AI" if is_ai else "Human",
        "is_ai": is_ai,
        "probability_ai": round(proba, 6),
        "confidence": round(conf, 6),
    }


def _predict(payload: dict) -> dict:
    """Prefer backend API when reachable; otherwise use local pipeline."""
    if backend_ok:
        try:
            return _predict_via_api(payload)
        except Exception:
            # Backend died between health check and predict — fall through
            pass
    return _predict_local(payload)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
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
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Inference mode banner — backend if reachable, else standalone
# ---------------------------------------------------------------------------

backend_ok = _check_backend()
local_pipeline = _load_local_pipeline()
local_ok = local_pipeline is not None

if not backend_ok and not local_ok:
    st.markdown(
        f"""<div class="error-box">
        ⚠ No inference backend available.<br>
        Backend unreachable at <strong>{API_BASE_URL or '(disabled)'}</strong>
        and no model found at <strong>{MODEL_PATH}</strong><br>
        Start the API: <code>uvicorn src.api:app --reload --port 8000</code>
        or train the model: <code>python -m src.train data/ai_writing_detection_dataset.csv {MODEL_PATH}</code>
        </div>""",
        unsafe_allow_html=True,
    )
    st.stop()

if backend_ok:
    st.caption("🔌 Connected to FastAPI backend")
else:
    st.caption("☁️ Running in standalone mode (local model)")

# ---------------------------------------------------------------------------
# Two-column layout: form (left) | results (right)
# ---------------------------------------------------------------------------

col_form, col_result = st.columns([1.1, 0.9], gap="large")

# ── LEFT: Input form ─────────────────────────────────────────────────────────

with col_form:

    # ── Behavioral telemetry ─────────────────────────────────
    st.markdown('<div class="section-label">Behavioral Telemetry</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    bt1, bt2 = st.columns(2)
    with bt1:
        editing_time = st.number_input(
            "Editing Time (min)", min_value=0.0, value=90.0, step=1.0,
            help="Total time spent editing the document in minutes"
        )
        revision_count = st.number_input(
            "Revision Count", min_value=0.0, value=5.0, step=1.0,
            help="Number of revision cycles recorded"
        )
    with bt2:
        typing_speed = st.number_input(
            "Typing Speed (WPM)", min_value=0.0, value=45.0, step=1.0,
            help="Average words per minute during composition"
        )
        submission_hour = st.number_input(
            "Submission Hour (0–23)", min_value=0, max_value=23, value=14,
            help="Hour of day the document was submitted"
        )

    # ── Linguistic metrics ────────────────────────────────────
    st.markdown('<div class="section-label">Linguistic Metrics</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    lm1, lm2 = st.columns(2)
    with lm1:
        word_count = st.number_input(
            "Word Count", min_value=0, value=800, step=10,
            help="Total number of words in the submission"
        )
        avg_sentence_len = st.number_input(
            "Avg Sentence Length", min_value=0.0, value=18.0, step=0.5,
            help="Mean number of words per sentence"
        )
        grammar_errors = st.number_input(
            "Grammar Errors", min_value=0.0, value=3.0, step=1.0,
            help="Number of grammar errors detected"
        )
        vocabulary_richness = st.slider(
            "Vocabulary Richness", min_value=0.0, max_value=1.0, value=0.6, step=0.01,
            help="Ratio of unique vocabulary (0 = repetitive, 1 = highly varied)"
        )
    with lm2:
        passive_voice = st.slider(
            "Passive Voice Ratio", min_value=0.0, max_value=1.0, value=0.2, step=0.01,
            help="Fraction of sentences using passive voice"
        )
        reading_level = st.number_input(
            "Reading Level (Flesch–Kincaid grade)", min_value=0.0, value=12.0, step=0.5,
            help="Flesch-Kincaid grade level score"
        )
        sentence_complexity = st.number_input(
            "Sentence Complexity", min_value=0.0, value=2.0, step=0.1,
            help="Composite syntactic complexity score"
        )
        punctuation_density = st.number_input(
            "Punctuation Density", min_value=0.0, value=0.07, step=0.01, format="%.3f",
            help="Punctuation marks per word"
        )

    # ── Document metadata ─────────────────────────────────────
    st.markdown('<div class="section-label">Document Metadata</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    dm1, dm2 = st.columns(2)
    with dm1:
        submission_type = st.selectbox("Submission Type", SUBMISSION_TYPES, index=0)
        academic_level  = st.selectbox("Academic Level",   ACADEMIC_LEVELS,   index=0)
        primary_language = st.selectbox("Primary Language", PRIMARY_LANGUAGES, index=0)
    with dm2:
        plagiarism_score = st.number_input(
            "Plagiarism Score (%)", min_value=0.0, value=1.5, step=0.1,
            help="Percentage similarity flagged by plagiarism checker"
        )
        citation_count = st.number_input(
            "Citation Count", min_value=0.0, value=4.0, step=1.0,
            help="Number of citations in the document"
        )
        font_family = st.selectbox("Font Family", FONT_FAMILIES, index=0)

    # ── Analyse button ────────────────────────────────────────
    st.markdown("<div style='margin-top:1.4rem;'>", unsafe_allow_html=True)
    analyse_clicked = st.button("Analyse Writing Sample", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── RIGHT: Results panel ──────────────────────────────────────────────────────

with col_result:

    if not analyse_clicked:
        st.markdown(
            """
            <div class="idle-box">
                &gt; awaiting sample...<br>
                &gt; fill in the form and click<br>
                &gt; "Analyse Writing Sample"<br>
                &gt; <span style="color:#38BDF8;">█</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        # Build the API payload
        payload = {
            "Submission_Type":       submission_type,
            "Academic_Level":        academic_level,
            "Primary_Language":      primary_language,
            "Font_Family":           font_family,
            "Word_Count":            int(word_count),
            "Submission_Hour":       int(submission_hour),
            "Average_Sentence_Length": float(avg_sentence_len),
            "Grammar_Errors":        float(grammar_errors),
            "Vocabulary_Richness":   float(vocabulary_richness),
            "Passive_Voice_Ratio":   float(passive_voice),
            "Reading_Level":         float(reading_level),
            "Editing_Time":          float(editing_time),
            "Revision_Count":        float(revision_count),
            "Typing_Speed":          float(typing_speed),
            "Plagiarism_Score":      float(plagiarism_score),
            "Citation_Count":        float(citation_count),
            "Punctuation_Density":   float(punctuation_density),
            "Sentence_Complexity":   float(sentence_complexity),
        }

        # Call inference (backend API preferred, local pipeline fallback)
        with st.spinner("Analysing..."):
            try:
                result = _predict(payload)
            except requests.exceptions.ConnectionError:
                try:
                    result = _predict_local(payload)
                except Exception:
                    st.markdown(
                        f'<div class="error-box">ERR: cannot connect to {API_BASE_URL}</div>',
                        unsafe_allow_html=True,
                    )
                    st.stop()
            except requests.exceptions.HTTPError as e:
                try:
                    result = _predict_local(payload)
                except Exception:
                    st.markdown(
                        f'<div class="error-box">ERR: API returned {e.response.status_code}</div>',
                        unsafe_allow_html=True,
                    )
                    st.stop()
            except RuntimeError as e:
                st.markdown(
                    f'<div class="error-box">ERR: {e}</div>',
                    unsafe_allow_html=True,
                )
                st.stop()

        is_ai      = result["is_ai"]
        label      = result["label"]
        prob_ai    = result["probability_ai"]
        confidence = result["confidence"]

        # ── Verdict terminal block ──────────────────────────
        verdict_class = "verdict-text-ai" if is_ai else "verdict-text-human"
        verdict_text  = "AI ASSISTED" if is_ai else "HUMAN AUTHORED"
        prob_pct      = f"{prob_ai * 100:.1f}"
        conf_pct      = f"{confidence * 100:.1f}"

        st.markdown(
            f"""
            <div class="verdict-terminal">
                <div class="verdict-prompt">$ writing-detector --analyse sample.txt</div>
                <div class="{verdict_class}">VERDICT: {verdict_text}<span class="verdict-cursor"></span></div>
                <div class="verdict-sub">
                    P(AI)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span>{prob_pct}%</span><br>
                    Confidence &nbsp;<span>{conf_pct}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Probability bar ─────────────────────────────────
        bar_class  = "prob-bar-fill-ai" if is_ai else "prob-bar-fill-human"
        bar_width  = f"{prob_ai * 100:.1f}%"
        st.markdown(
            f"""
            <div style="margin-bottom:1rem;">
                <div style="font-size:0.72rem;color:#94A3B8;
                            font-family:'JetBrains Mono',monospace;
                            margin-bottom:0.3rem;">AI PROBABILITY</div>
                <div class="prob-bar-container">
                    <div class="{bar_class}" style="width:{bar_width};"></div>
                </div>
                <div class="prob-labels">
                    <span>0% Human</span><span>100% AI</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Metric cards ─────────────────────────────────────
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-value">{prob_pct}%</div>
                    <div class="metric-label">P(AI)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{100 - float(prob_pct):.1f}%</div>
                    <div class="metric-label">P(Human)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{conf_pct}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Feature snapshot table ────────────────────────────
        st.markdown(
            '<div class="section-label" style="margin-top:0;">Feature Snapshot</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        import pandas as pd

        snapshot_data = {
            "Feature": [
                "Editing Time", "Typing Speed", "Revision Count",
                "Word Count", "Grammar Errors", "Vocab Richness",
                "Plagiarism Score", "Reading Level",
            ],
            "Value": [
                f"{editing_time:.0f} min", f"{typing_speed:.0f} wpm",
                f"{revision_count:.0f}", f"{word_count}",
                f"{grammar_errors:.0f}", f"{vocabulary_richness:.2f}",
                f"{plagiarism_score:.1f}%", f"{reading_level:.1f}",
            ],
        }
        df_snapshot = pd.DataFrame(snapshot_data)

        st.dataframe(
            df_snapshot,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature": st.column_config.TextColumn("Feature", width="medium"),
                "Value":   st.column_config.TextColumn("Value",   width="small"),
            },
        )

        # ── Raw JSON expander ─────────────────────────────────
        with st.expander("Raw API response", expanded=False):
            st.json(result)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #2D3748;
                text-align:center;font-size:0.72rem;color:#4A5568;
                font-family:'JetBrains Mono',monospace;">
        AI Writing Detector &nbsp;·&nbsp; XGBoost + FastAPI + Streamlit
        &nbsp;·&nbsp; Kaggle Dataset: Razan Ihab Abdellatif
    </div>
    """,
    unsafe_allow_html=True,
)
