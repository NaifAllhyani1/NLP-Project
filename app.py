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
# Theme State
# ─────────────────────────────────────────────

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True


# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────

def inject_styles(dark: bool):
    if dark:
        palette = """
            --bg-base:        #0d1117;
            --bg-surface:     #161b22;
            --bg-card:        #1c2330;
            --bg-input:       #161b22;
            --border:         #30363d;
            --border-subtle:  #21262d;
            --text-primary:   #e6edf3;
            --text-secondary: #8b949e;
            --text-muted:     #484f58;
            --accent-green:   #3fb950;
            --accent-red:     #f85149;
            --accent-blue:    #58a6ff;
            --accent-orange:  #d29922;
            --accent-glow-g:  rgba(63, 185, 80, 0.15);
            --accent-glow-r:  rgba(248, 81, 73, 0.15);
            --shadow:         0 8px 32px rgba(0,0,0,0.4);
            --shadow-sm:      0 2px 8px rgba(0,0,0,0.3);
        """
    else:
        palette = """
            --bg-base:        #f5f7fa;
            --bg-surface:     #ffffff;
            --bg-card:        #ffffff;
            --bg-input:       #f0f3f7;
            --border:         #d0d7de;
            --border-subtle:  #e8ecf0;
            --text-primary:   #1c2128;
            --text-secondary: #57606a;
            --text-muted:     #8c959f;
            --accent-green:   #1a7f37;
            --accent-red:     #cf222e;
            --accent-blue:    #0969da;
            --accent-orange:  #9a6700;
            --accent-glow-g:  rgba(26, 127, 55, 0.08);
            --accent-glow-r:  rgba(207, 34, 46, 0.08);
            --shadow:         0 8px 32px rgba(0,0,0,0.08);
            --shadow-sm:      0 2px 8px rgba(0,0,0,0.06);
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Geist:wght@300;400;500;600&display=swap');

        :root {{
            {palette}
        }}

        /* ── Reset & Base ── */
        html, body, [class*="css"],
        .stApp, .block-container,
        section[data-testid="stSidebar"] {{
            font-family: 'Geist', sans-serif !important;
            background-color: var(--bg-base) !important;
            color: var(--text-primary) !important;
        }}

        .block-container {{
            padding: 2rem 2.5rem 4rem !important;
            max-width: 1100px !important;
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background-color: var(--bg-surface) !important;
            border-right: 1px solid var(--border) !important;
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding: 2rem 1.5rem !important;
        }}

        /* ── Header Banner ── */
        .app-header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 2rem 2.5rem;
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-sm);
            position: relative;
            overflow: hidden;
        }}
        .app-header::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
        }}
        .app-header h1 {{
            font-size: 1.6rem;
            font-weight: 600;
            margin: 0 0 0.35rem;
            letter-spacing: -0.02em;
            color: var(--text-primary);
        }}
        .app-header p {{
            margin: 0;
            font-size: 0.82rem;
            color: var(--text-secondary);
            font-family: 'Geist Mono', monospace;
            letter-spacing: 0.01em;
        }}
        .header-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.25rem 0.65rem;
            font-family: 'Geist Mono', monospace;
            font-size: 0.7rem;
            color: var(--text-secondary);
            white-space: nowrap;
        }}
        .header-badge .dot {{
            width: 6px; height: 6px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
        }}

        /* ── Input Section ── */
        .stTextArea textarea {{
            background: var(--bg-input) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
            font-family: 'Geist', sans-serif !important;
            font-size: 0.9rem !important;
            resize: vertical !important;
            transition: border-color 0.2s !important;
        }}
        .stTextArea textarea:focus {{
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
            outline: none !important;
        }}
        .stTextArea label {{
            color: var(--text-secondary) !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
        }}

        /* ── Selectbox ── */
        .stSelectbox label {{
            color: var(--text-secondary) !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
        }}
        .stSelectbox > div > div {{
            background: var(--bg-input) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }}

        /* ── Buttons ── */
        .stButton > button[kind="primary"] {{
            background: var(--text-primary) !important;
            color: var(--bg-base) !important;
            border: none !important;
            border-radius: 8px !important;
            font-family: 'Geist', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.02em !important;
            padding: 0.6rem 1.2rem !important;
            transition: opacity 0.15s !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            opacity: 0.85 !important;
        }}
        .stButton > button:not([kind="primary"]) {{
            background: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            font-family: 'Geist', sans-serif !important;
            font-weight: 500 !important;
            font-size: 0.82rem !important;
            transition: border-color 0.15s, background 0.15s !important;
        }}
        .stButton > button:not([kind="primary"]):hover {{
            border-color: var(--accent-blue) !important;
            background: var(--bg-input) !important;
        }}

        /* ── Result Cards ── */
        .result-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.75rem 1.5rem;
            text-align: center;
            transition: border-color 0.25s, box-shadow 0.25s;
            box-shadow: var(--shadow-sm);
            height: 100%;
        }}
        .result-card:hover {{
            box-shadow: var(--shadow);
        }}
        .result-card.positive {{
            border-color: var(--accent-green);
            background: linear-gradient(180deg, var(--accent-glow-g), var(--bg-card));
        }}
        .result-card.negative {{
            border-color: var(--accent-red);
            background: linear-gradient(180deg, var(--accent-glow-r), var(--bg-card));
        }}
        .result-card .model-name {{
            font-family: 'Geist Mono', monospace;
            font-size: 0.68rem;
            color: var(--text-muted);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }}
        .result-card .sentiment-label {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            letter-spacing: -0.01em;
        }}
        .result-card .sentiment-label.positive {{ color: var(--accent-green); }}
        .result-card .sentiment-label.negative {{ color: var(--accent-red); }}
        .result-card .confidence {{
            font-family: 'Geist Mono', monospace;
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin-bottom: 0.35rem;
        }}
        .result-card .latency {{
            font-family: 'Geist Mono', monospace;
            font-size: 0.68rem;
            color: var(--text-muted);
        }}

        /* ── Metric Boxes ── */
        [data-testid="metric-container"] {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            padding: 1.25rem 1.5rem !important;
        }}
        [data-testid="metric-container"] label {{
            color: var(--text-muted) !important;
            font-size: 0.72rem !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
        }}
        [data-testid="metric-container"] [data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
        }}

        /* ── Alerts / Banners ── */
        .stAlert {{
            border-radius: 8px !important;
            border-width: 1px !important;
        }}
        .stSuccess {{
            background: var(--accent-glow-g) !important;
            border-color: var(--accent-green) !important;
            color: var(--text-primary) !important;
        }}
        .stError {{
            background: var(--accent-glow-r) !important;
            border-color: var(--accent-red) !important;
            color: var(--text-primary) !important;
        }}

        /* ── Divider ── */
        hr {{
            border: none !important;
            border-top: 1px solid var(--border-subtle) !important;
            margin: 1.5rem 0 !important;
        }}

        /* ── Spinner ── */
        .stSpinner > div {{
            border-color: var(--accent-blue) transparent transparent transparent !important;
        }}

        /* ── Toggle ── */
        .stToggle label {{
            color: var(--text-secondary) !important;
            font-size: 0.85rem !important;
        }}

        /* ── Expander ── */
        .streamlit-expanderHeader {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-secondary) !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
        }}
        .streamlit-expanderContent {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px !important;
        }}

        /* ── Info Box ── */
        .stInfo {{
            background: rgba(88,166,255,0.08) !important;
            border-color: var(--accent-blue) !important;
            color: var(--text-primary) !important;
            border-radius: 8px !important;
        }}

        /* ── Sidebar model pills ── */
        .model-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.3rem 0.7rem;
            font-size: 0.73rem;
            color: var(--text-secondary);
            font-family: 'Geist Mono', monospace;
            margin: 3px 2px;
            letter-spacing: 0.03em;
        }}

        /* ── Section label ── */
        .section-label {{
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
        }}

        /* ── Consensus card ── */
        .consensus-card {{
            display: flex;
            align-items: center;
            gap: 1rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem 1.75rem;
            margin-top: 1.5rem;
        }}
        .consensus-card .verdict {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }}
        .consensus-card .tally {{
            font-family: 'Geist Mono', monospace;
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }}
        .consensus-icon {{
            font-size: 1.6rem;
            flex-shrink: 0;
        }}

        /* ── Dataframe ── */
        .stDataFrame {{
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }}

        /* ── Responsive tweaks ── */
        @media (max-width: 768px) {{
            .block-container {{
                padding: 1rem 1rem 3rem !important;
            }}
            .app-header {{
                flex-direction: column;
                padding: 1.5rem 1.25rem;
            }}
            .app-header h1 {{
                font-size: 1.3rem;
            }}
        }}

        /* ── Streamlit top bar hide ── */
        #MainMenu, footer, header[data-testid="stHeader"] {{
            display: none !important;
        }}
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

LABEL_EMOJI = {"POSITIVE": "↑", "NEGATIVE": "↓"}


# ─────────────────────────────────────────────
# Label Unification
# ─────────────────────────────────────────────

def unify_label(raw_label: str, model_name: str) -> str:
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
    pipe   = load_pipeline(model_name)
    start  = time.time()
    result = pipe(text)[0]
    elapsed = time.time() - start

    unified    = unify_label(result["label"], model_name)
    confidence = round(result["score"] * 100, 2)
    return unified, confidence, elapsed


# ─────────────────────────────────────────────
# Render a single result card
# ─────────────────────────────────────────────

def render_result_card(
    model_name: str,
    label: str,
    confidence: float,
    elapsed: float,
) -> None:
    css_class = label.lower()
    emoji     = LABEL_EMOJI[label]

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="model-name">{model_name}</div>
            <div class="sentiment-label {css_class}">{emoji} {label}</div>
            <div class="confidence">{confidence:.2f}%</div>
            <div class="latency">⏱ {elapsed * 1000:.0f} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Inject styles based on current theme
# ─────────────────────────────────────────────

inject_styles(st.session_state.dark_mode)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    # ── Theme Toggle ──────────────────────────
    col_logo, col_theme = st.columns([3, 1])

    with col_logo:
        st.markdown(
            "<p style='font-weight:600;font-size:0.95rem;margin:0;'>Sentiment Classifier</p>"
            "<p style='color:var(--text-muted);font-size:0.72rem;font-family:Geist Mono,monospace;margin:0;'>v2.0</p>",
            unsafe_allow_html=True,
        )

    with col_theme:
        theme_label = "☀️" if st.session_state.dark_mode else "🌙"
        if st.button(theme_label, key="theme_toggle", help="Toggle light / dark mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown("---")

    # ── Models ────────────────────────────────
    st.markdown('<p class="section-label">Models</p>', unsafe_allow_html=True)

    model_meta = [
        ("DistilBERT (SST-2)", "distilbert", "English"),
        ("RoBERTa (Twitter)",  "roberta",    "Social"),
        ("BERT Multilingual",  "bert-multi", "Multilingual"),
    ]
    for name, tag, scope in model_meta:
        st.markdown(
            f'<span class="model-pill">◈ {name}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Label Mapping ─────────────────────────
    st.markdown('<p class="section-label">Label Mapping</p>', unsafe_allow_html=True)
    st.markdown(
        """
        | Model | Raw → Unified |
        |---|---|
        | DistilBERT | POS / NEG |
        | RoBERTa | L0,L1→NEG · L2→POS |
        | BERT | 1–3★→NEG · 4–5★→POS |
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    compare_mode = st.toggle(
        "Compare all models",
        value=False,
        help="Run all three models simultaneously and display results in columns.",
    )

    st.markdown("---")
    st.caption("Models are loaded once and cached per session.")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <div>
            <h1>Sentiment Analysis</h1>
            <p>DistilBERT · RoBERTa · BERT Multilingual — Hugging Face Transformers</p>
        </div>
        <div style="display:flex;flex-direction:column;gap:0.4rem;align-items:flex-end;flex-shrink:0;">
            <span class="header-badge"><span class="dot"></span>Live Inference</span>
            <span class="header-badge">3 Models Available</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────

user_text = st.text_area(
    label="Text to analyse",
    placeholder="e.g.  'The movie was absolutely wonderful — I loved every minute of it!'",
    height=110,
    key="user_text_input",
)

col_select, col_btn = st.columns([3, 1])

with col_select:
    if not compare_mode:
        selected_display = st.selectbox(
            "Model",
            options=list(MODEL_OPTIONS.keys()),
            key="model_selector",
        )
        selected_model = MODEL_OPTIONS[selected_display]
    else:
        st.info("Compare mode — all three models will run on submit.")

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button(
        "Run →",
        type="primary",
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────

if predict_clicked:

    if not user_text or not user_text.strip():
        st.warning("Please enter some text before clicking Run.")
        st.stop()

    if len(user_text.strip()) < 3:
        st.warning("Input is too short for meaningful sentiment analysis.")
        st.stop()

    st.markdown("---")

    # ── Single-model mode ─────────────────────
    if not compare_mode:
        with st.spinner(f"Running {selected_model}…"):
            try:
                label, confidence, elapsed = predict(user_text, selected_model)
            except RuntimeError as err:
                st.error(f"Model error: {err}")
                st.stop()

        m1, m2, m3 = st.columns(3)
        m1.metric("Model",      selected_model)
        m2.metric("Sentiment",  f"{LABEL_EMOJI[label]} {label}")
        m3.metric("Confidence", f"{confidence:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        if label == "POSITIVE":
            st.success(
                f"**{selected_model}** → **POSITIVE** with **{confidence:.2f}%** confidence "
                f"· {elapsed * 1000:.0f} ms"
            )
        else:
            st.error(
                f"**{selected_model}** → **NEGATIVE** with **{confidence:.2f}%** confidence "
                f"· {elapsed * 1000:.0f} ms"
            )

        with st.expander("Raw prediction details"):
            pipe = load_pipeline(selected_model)
            raw  = pipe(user_text)[0]
            st.json(
                {
                    "model"          : MODEL_CONFIG[selected_model],
                    "raw_label"      : raw["label"],
                    "raw_score"      : round(raw["score"], 6),
                    "unified_label"  : label,
                    "confidence_pct" : confidence,
                    "inference_ms"   : round(elapsed * 1000, 1),
                }
            )

    # ── Compare-all mode ──────────────────────
    else:
        st.markdown(
            '<p class="section-label" style="margin-bottom:1rem;">All-Model Results</p>',
            unsafe_allow_html=True,
        )

        model_keys = list(MODEL_CONFIG.keys())
        cols       = st.columns(len(model_keys), gap="medium")
        results    = {}

        for col, model_name in zip(cols, model_keys):
            with col:
                with st.spinner(f"{model_name}…"):
                    try:
                        label, confidence, elapsed = predict(user_text, model_name)
                        results[model_name] = (label, confidence, elapsed)
                        render_result_card(model_name, label, confidence, elapsed)
                    except RuntimeError as err:
                        st.error(f"{model_name} failed:\n{err}")

        # ── Consensus ─────────────────────────
        if results:
            labels_list = [v[0] for v in results.values()]
            pos_count   = labels_list.count("POSITIVE")
            neg_count   = labels_list.count("NEGATIVE")
            total       = len(model_keys)

            if pos_count == total:
                icon    = "✓"
                verdict = "Unanimous — POSITIVE"
                color   = "var(--accent-green)"
            elif neg_count == total:
                icon    = "✗"
                verdict = "Unanimous — NEGATIVE"
                color   = "var(--accent-red)"
            else:
                icon    = "≈"
                majority = "POSITIVE" if pos_count > neg_count else "NEGATIVE"
                verdict = f"Split — majority {majority}"
                color   = "var(--accent-orange)"

            st.markdown(
                f"""
                <div class="consensus-card">
                    <div class="consensus-icon" style="color:{color};">{icon}</div>
                    <div>
                        <div class="verdict">{verdict}</div>
                        <div class="tally">{pos_count} positive · {neg_count} negative across {total} models</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Comparison table"):
                import pandas as pd

                table = pd.DataFrame(
                    [
                        {
                            "Model"        : name,
                            "Model ID"     : MODEL_CONFIG[name],
                            "Sentiment"    : f"{LABEL_EMOJI[lbl]} {lbl}",
                            "Confidence"   : f"{conf:.2f}%",
                            "Latency (ms)" : f"{elap * 1000:.0f}",
                        }
                        for name, (lbl, conf, elap) in results.items()
                    ]
                )
                st.dataframe(table, use_container_width=True, hide_index=True)