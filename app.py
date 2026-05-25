"""
app.py
======
Interactive Streamlit web application for live sentiment classification.
Supports single-model prediction and a side-by-side "Compare All" mode.

Run with:
  streamlit run app.py
"""

import time
import streamlit as st
from transformers import pipeline

# ─────────────────────────────────────────────
# Page Configuration  (must be the first st call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Sentiment Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
        /* ── Global typography ── */
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        /* ── Header banner ── */
        .app-header {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            border-radius: 12px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            border-left: 5px solid #00d4aa;
        }
        .app-header h1 { color: #ffffff; font-size: 2rem; margin: 0; }
        .app-header p  { color: #a8c5d8; margin: 0.4rem 0 0; font-size: 0.95rem; }

        /* ── Result cards ── */
        .result-card {
            background: #1e2d3d;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #2d4a5e;
            transition: border-color 0.3s;
        }
        .result-card.positive { border-color: #00d4aa; }
        .result-card.negative { border-color: #ff6b6b; }

        .result-card .model-name {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem;
            color: #7a9ab5;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }
        .result-card .sentiment-label {
            font-size: 1.7rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .result-card .sentiment-label.positive { color: #00d4aa; }
        .result-card .sentiment-label.negative { color: #ff6b6b; }
        .result-card .confidence {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1rem;
            color: #c5d8e8;
        }

        /* ── Model pill badges in sidebar ── */
        .model-pill {
            display: inline-block;
            background: #1e2d3d;
            border: 1px solid #2d4a5e;
            border-radius: 20px;
            padding: 2px 10px;
            font-size: 0.75rem;
            color: #a8c5d8;
            font-family: 'IBM Plex Mono', monospace;
            margin: 2px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

MODEL_OPTIONS = {
    "DistilBERT (SST-2)"        : "DistilBERT",
    "RoBERTa (Twitter)"         : "RoBERTa",
    "BERT Multilingual (Stars)" : "BERT",
}

MODEL_CONFIG = {
    "DistilBERT": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa"   : "cardiffnlp/twitter-roberta-base-sentiment",
    "BERT"      : "nlptown/bert-base-multilingual-uncased-sentiment",
}

LABEL_EMOJI = {"POSITIVE": "✅", "NEGATIVE": "❌"}


# ─────────────────────────────────────────────
# Label Unification  (mirrors evaluate_models.py exactly)
# ─────────────────────────────────────────────

def unify_label(raw_label: str, model_name: str) -> str:
    """
    Normalise model-specific output labels to 'POSITIVE' or 'NEGATIVE'.

    DistilBERT → already 'POSITIVE' / 'NEGATIVE'.
    RoBERTa    → LABEL_0 = NEGATIVE, LABEL_1 = NEGATIVE, LABEL_2 = POSITIVE.
    BERT       → star rating: 1–3 stars = NEGATIVE, 4–5 stars = POSITIVE.

    Raises ValueError for unexpected label formats.
    """
    label = raw_label.strip().upper()

    if model_name == "DistilBERT":
        return label

    elif model_name == "RoBERTa":
        mapping = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "NEGATIVE",
            "LABEL_2": "POSITIVE",
        }
        if label not in mapping:
            raise ValueError(f"Unexpected RoBERTa label: '{raw_label}'")
        return mapping[label]

    elif model_name == "BERT":
        digit_str = label.split()[0]
        if not digit_str.isdigit():
            raise ValueError(f"Unexpected BERT label: '{raw_label}'")
        return "POSITIVE" if int(digit_str) >= 4 else "NEGATIVE"

    else:
        raise ValueError(f"Unknown model name: '{model_name}'")


# ─────────────────────────────────────────────
# Cached Model Loading
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline(model_name: str):
    """
    Load and cache a HuggingFace pipeline.
    `@st.cache_resource` ensures each model is downloaded once per session.

    Parameters
    ----------
    model_name : str  Key from MODEL_CONFIG ('DistilBERT', 'RoBERTa', 'BERT').

    Returns
    -------
    HuggingFace pipeline object or raises RuntimeError.
    """
    model_id = MODEL_CONFIG[model_name]
    try:
        return pipeline(
            task="text-classification",
            model=model_id,
            truncation=True,
            max_length=512,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{model_name}' ({model_id}).\n"
            f"Check your internet connection or model ID.\nDetails: {exc}"
        )


# ─────────────────────────────────────────────
# Prediction Helper
# ─────────────────────────────────────────────

def predict(text: str, model_name: str) -> tuple[str, float, float]:
    """
    Run inference for a single text string.

    Parameters
    ----------
    text       : User-supplied sentence.
    model_name : One of 'DistilBERT', 'RoBERTa', 'BERT'.

    Returns
    -------
    (unified_label, confidence_pct, elapsed_seconds)
    """
    pipe  = load_pipeline(model_name)
    start = time.time()
    result = pipe(text)[0]            # returns {'label': ..., 'score': ...}
    elapsed = time.time() - start

    unified    = unify_label(result["label"], model_name)
    confidence = round(result["score"] * 100, 2)
    return unified, confidence, elapsed


# ─────────────────────────────────────────────
# Render a single result card (HTML)
# ─────────────────────────────────────────────

def render_result_card(
    model_name: str,
    label: str,
    confidence: float,
    elapsed: float,
) -> None:
    """Inject an HTML result card into the Streamlit layout."""
    css_class = label.lower()    # 'positive' or 'negative'
    emoji     = LABEL_EMOJI[label]

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="model-name">{model_name}</div>
            <div class="sentiment-label {css_class}">{emoji} {label}</div>
            <div class="confidence">{confidence:.2f}% confidence</div>
            <div style="font-size:0.72rem;color:#5a7a9a;margin-top:0.4rem;">
                ⏱ {elapsed * 1000:.0f} ms
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 Sentiment Classifier")
    st.markdown("---")

    st.markdown("**Available models:**")
    for display, key in MODEL_OPTIONS.items():
        st.markdown(
            f'<span class="model-pill">{display}</span>', unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        """
        **Label mapping**
        | Model | Raw Label → Unified |
        |---|---|
        | DistilBERT | POSITIVE / NEGATIVE |
        | RoBERTa | LABEL_0→NEG, LABEL_2→POS |
        | BERT | 1–3★→NEG, 4–5★→POS |
        """
    )

    st.markdown("---")
    compare_mode = st.toggle(
        "🔀 Compare all models side-by-side",
        value=False,
        help="Run all three models simultaneously and display results in columns.",
    )

    st.markdown("---")
    st.caption("Models are loaded once and cached for the session.")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <h1>🧠 Sentiment Analysis Dashboard</h1>
        <p>Powered by DistilBERT · RoBERTa · BERT Multilingual via Hugging Face Transformers</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────

user_text = st.text_area(
    label="Enter text to analyse:",
    placeholder=(
        "e.g.  'The movie was absolutely wonderful — I loved every minute of it!'"
    ),
    height=120,
    key="user_text_input",
)

col_select, col_btn = st.columns([3, 1])

with col_select:
    if not compare_mode:
        selected_display = st.selectbox(
            "Select a model:",
            options=list(MODEL_OPTIONS.keys()),
            key="model_selector",
        )
        selected_model = MODEL_OPTIONS[selected_display]
    else:
        st.info("ℹ️ **Compare mode active** — all three models will run on submit.", icon="🔀")

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)   # vertical alignment spacer
    predict_clicked = st.button(
        "Predict ▶",
        type="primary",
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────

if predict_clicked:

    # ── Validation ────────────────────────────
    if not user_text or not user_text.strip():
        st.warning("⚠️ Please enter some text before clicking Predict.")
        st.stop()

    if len(user_text.strip()) < 3:
        st.warning("⚠️ Input is too short for meaningful sentiment analysis.")
        st.stop()

    st.markdown("---")

    # ── Single-model mode ─────────────────────
    if not compare_mode:
        with st.spinner(f"Running {selected_model} inference …"):
            try:
                label, confidence, elapsed = predict(user_text, selected_model)
            except RuntimeError as err:
                st.error(f"Model error: {err}")
                st.stop()

        # Metric display.
        m1, m2, m3 = st.columns(3)
        m1.metric("Model",       selected_model)
        m2.metric("Sentiment",   f"{LABEL_EMOJI[label]} {label}")
        m3.metric("Confidence",  f"{confidence:.2f}%")

        # Success / error banner.
        if label == "POSITIVE":
            st.success(
                f"**{selected_model}** predicts **POSITIVE** sentiment "
                f"with **{confidence:.2f}%** confidence."
            )
        else:
            st.error(
                f"**{selected_model}** predicts **NEGATIVE** sentiment "
                f"with **{confidence:.2f}%** confidence."
            )

        # Expandable debug info.
        with st.expander("🔍 Raw prediction details"):
            pipe  = load_pipeline(selected_model)
            raw   = pipe(user_text)[0]
            st.json(
                {
                    "model"           : MODEL_CONFIG[selected_model],
                    "raw_label"       : raw["label"],
                    "raw_score"       : round(raw["score"], 6),
                    "unified_label"   : label,
                    "confidence_pct"  : confidence,
                    "inference_ms"    : round(elapsed * 1000, 1),
                }
            )

    # ── Compare-all mode ──────────────────────
    else:
        st.subheader("📊 All-Model Comparison")

        model_keys = list(MODEL_CONFIG.keys())   # ['DistilBERT', 'RoBERTa', 'BERT']
        cols       = st.columns(len(model_keys))
        results    = {}

        for col, model_name in zip(cols, model_keys):
            with col:
                with st.spinner(f"Loading {model_name} …"):
                    try:
                        label, confidence, elapsed = predict(user_text, model_name)
                        results[model_name] = (label, confidence, elapsed)
                        render_result_card(model_name, label, confidence, elapsed)
                    except RuntimeError as err:
                        st.error(f"{model_name} failed:\n{err}")

        # ── Consensus badge ───────────────────
        if results:
            st.markdown("---")
            labels_list = [v[0] for v in results.values()]
            pos_count   = labels_list.count("POSITIVE")
            neg_count   = labels_list.count("NEGATIVE")

            if pos_count == len(model_keys):
                st.success(
                    f"✅ **Unanimous consensus: POSITIVE** — all {len(model_keys)} models agree."
                )
            elif neg_count == len(model_keys):
                st.error(
                    f"❌ **Unanimous consensus: NEGATIVE** — all {len(model_keys)} models agree."
                )
            else:
                majority = "POSITIVE" if pos_count > neg_count else "NEGATIVE"
                st.warning(
                    f"⚖️ **Mixed results** — majority vote: **{majority}** "
                    f"({pos_count} POSITIVE / {neg_count} NEGATIVE)."
                )

            # Summary table.
            with st.expander("📋 Comparison summary table"):
                import pandas as pd

                table = pd.DataFrame(
                    [
                        {
                            "Model"      : name,
                            "Model ID"   : MODEL_CONFIG[name],
                            "Sentiment"  : f"{LABEL_EMOJI[lbl]} {lbl}",
                            "Confidence" : f"{conf:.2f}%",
                            "Latency (ms)": f"{elap * 1000:.0f}",
                        }
                        for name, (lbl, conf, elap) in results.items()
                    ]
                )
                st.dataframe(table, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("---")
st.caption(
    "University Group Assignment · NLP Sentiment Classification · "
    "Powered by 🤗 Hugging Face Transformers & Streamlit"
)