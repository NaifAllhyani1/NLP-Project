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
# Page Configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Sentiment Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
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
            --accent-glow-g:  rgba(63, 185, 80, 0.12);
            --accent-glow-r:  rgba(248, 81, 73, 0.12);
            --shadow:         0 8px 32px rgba(0,0,0,0.4);
            --shadow-sm:      0 2px 8px rgba(0,0,0,0.3);
            --nav-bg:         #161b22;
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
            --accent-glow-g:  rgba(26, 127, 55, 0.07);
            --accent-glow-r:  rgba(207, 34, 46, 0.07);
            --shadow:         0 8px 32px rgba(0,0,0,0.08);
            --shadow-sm:      0 2px 8px rgba(0,0,0,0.05);
            --nav-bg:         #ffffff;
        """

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Geist:wght@300;400;500;600&display=swap');

        :root {{ {palette} }}

        /* ── Collapse sidebar arrow entirely ── */
        [data-testid="collapsedControl"] {{ display: none !important; }}
        section[data-testid="stSidebar"] {{ display: none !important; }}

        /* ── Base ── */
        html, body, [class*="css"], .stApp, .block-container {{
            font-family: 'Geist', sans-serif !important;
            background-color: var(--bg-base) !important;
            color: var(--text-primary) !important;
        }}

        /* ── Wipe Streamlit chrome ── */
        #MainMenu, footer, header[data-testid="stHeader"] {{
            display: none !important;
        }}

        /* ── Main content area ── */
        .block-container {{
            padding: 0 !important;
            max-width: 100% !important;
        }}

        /* ── Navbar ── */
        .navbar {{
            position: sticky;
            top: 0;
            z-index: 999;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2.5rem;
            height: 56px;
            background: var(--nav-bg);
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(8px);
        }}
        .navbar-left {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}
        .navbar-brand {{
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.01em;
        }}
        .navbar-version {{
            font-family: 'Geist Mono', monospace;
            font-size: 0.65rem;
            color: var(--text-muted);
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 2px 6px;
        }}
        .navbar-right {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        .live-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-family: 'Geist Mono', monospace;
            font-size: 0.68rem;
            color: var(--text-secondary);
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 3px 10px;
        }}
        .live-dot {{
            width: 5px; height: 5px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.35; }}
        }}

        /* ── Page body wrapper ── */
        .page-body {{
            max-width: 860px;
            margin: 0 auto;
            padding: 2.5rem 2rem 5rem;
        }}

        /* ── Page title ── */
        .page-title {{
            margin-bottom: 2rem;
        }}
        .page-title h1 {{
            font-size: 1.75rem;
            font-weight: 600;
            letter-spacing: -0.03em;
            color: var(--text-primary);
            margin: 0 0 0.3rem;
        }}
        .page-title p {{
            font-family: 'Geist Mono', monospace;
            font-size: 0.75rem;
            color: var(--text-muted);
            margin: 0;
        }}

        /* ── Input card ── */
        .input-card {{
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--shadow-sm);
        }}

        /* ── TextArea ── */
        .stTextArea textarea {{
            background: var(--bg-input) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
            font-family: 'Geist', sans-serif !important;
            font-size: 0.9rem !important;
            resize: vertical !important;
            transition: border-color 0.2s !important;
            padding: 0.75rem 1rem !important;
        }}
        .stTextArea textarea:focus {{
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
            outline: none !important;
        }}
        .stTextArea label {{
            color: var(--text-secondary) !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
        }}

        /* ── Selectbox ── */
        .stSelectbox label {{
            color: var(--text-secondary) !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
        }}
        .stSelectbox > div > div {{
            background: var(--bg-input) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }}

        /* ── Toggle ── */
        .stToggle label {{
            color: var(--text-secondary) !important;
            font-size: 0.85rem !important;
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
            transition: opacity 0.15s !important;
        }}
        .stButton > button[kind="primary"]:hover {{ opacity: 0.82 !important; }}

        /* theme toggle button */
        .stButton > button:not([kind="primary"]) {{
            background: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            font-family: 'Geist', sans-serif !important;
            font-size: 0.82rem !important;
            transition: border-color 0.15s !important;
            padding: 0.3rem 0.75rem !important;
        }}
        .stButton > button:not([kind="primary"]):hover {{
            border-color: var(--text-secondary) !important;
        }}

        /* ── Metric boxes ── */
        [data-testid="metric-container"] {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            padding: 1.1rem 1.4rem !important;
        }}
        [data-testid="metric-container"] label {{
            color: var(--text-muted) !important;
            font-size: 0.7rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.09em !important;
        }}
        [data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
            font-size: 1.15rem !important;
            font-weight: 600 !important;
        }}

        /* ── Result cards ── */
        .result-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.75rem 1.25rem;
            text-align: center;
            box-shadow: var(--shadow-sm);
            transition: box-shadow 0.2s;
        }}
        .result-card:hover {{ box-shadow: var(--shadow); }}
        .result-card.positive {{
            border-color: var(--accent-green);
            background: linear-gradient(180deg, var(--accent-glow-g), var(--bg-card) 60%);
        }}
        .result-card.negative {{
            border-color: var(--accent-red);
            background: linear-gradient(180deg, var(--accent-glow-r), var(--bg-card) 60%);
        }}
        .result-card .model-name {{
            font-family: 'Geist Mono', monospace;
            font-size: 0.65rem;
            color: var(--text-muted);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }}
        .result-card .sentiment-label {{
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 0.45rem;
        }}
        .result-card .sentiment-label.positive {{ color: var(--accent-green); }}
        .result-card .sentiment-label.negative {{ color: var(--accent-red); }}
        .result-card .confidence {{
            font-family: 'Geist Mono', monospace;
            font-size: 0.88rem;
            color: var(--text-secondary);
            margin-bottom: 0.3rem;
        }}
        .result-card .latency {{
            font-family: 'Geist Mono', monospace;
            font-size: 0.65rem;
            color: var(--text-muted);
        }}

        /* ── Alerts ── */
        .stAlert {{ border-radius: 8px !important; }}
        .stSuccess {{
            background: var(--accent-glow-g) !important;
            border-color: var(--accent-green) !important;
        }}
        .stError {{
            background: var(--accent-glow-r) !important;
            border-color: var(--accent-red) !important;
        }}
        .stInfo {{
            background: rgba(88,166,255,0.07) !important;
            border-color: var(--accent-blue) !important;
            border-radius: 8px !important;
        }}

        /* ── Expander ── */
        .streamlit-expanderHeader {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-secondary) !important;
            font-size: 0.8rem !important;
        }}
        .streamlit-expanderContent {{
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px !important;
        }}

        /* ── Consensus card ── */
        .consensus-card {{
            display: flex;
            align-items: center;
            gap: 1rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            margin-top: 1.25rem;
        }}
        .consensus-icon {{ font-size: 1.5rem; flex-shrink: 0; }}
        .consensus-verdict {{ font-size: 1rem; font-weight: 600; color: var(--text-primary); }}
        .consensus-tally {{
            font-family: 'Geist Mono', monospace;
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
        }}

        /* ── Info section (models + mapping) ── */
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 2.5rem;
        }}
        .info-panel {{
            background: var(--bg-surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
        }}
        .info-panel-title {{
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }}
        .model-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-subtle);
        }}
        .model-row:last-child {{ border-bottom: none; }}
        .model-row-name {{
            font-size: 0.82rem;
            color: var(--text-primary);
            font-weight: 500;
        }}
        .model-row-scope {{
            font-family: 'Geist Mono', monospace;
            font-size: 0.68rem;
            color: var(--text-muted);
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 2px 7px;
        }}
        .mapping-row {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-subtle);
        }}
        .mapping-row:last-child {{ border-bottom: none; }}
        .mapping-model {{
            font-size: 0.8rem;
            font-weight: 500;
            color: var(--text-primary);
            white-space: nowrap;
        }}
        .mapping-rule {{
            font-family: 'Geist Mono', monospace;
            font-size: 0.68rem;
            color: var(--text-secondary);
            text-align: right;
        }}

        /* ── Divider ── */
        hr {{
            border: none !important;
            border-top: 1px solid var(--border-subtle) !important;
            margin: 1.25rem 0 !important;
        }}

        /* ── Responsive ── */
        @media (max-width: 700px) {{
            .navbar {{ padding: 0 1rem; }}
            .page-body {{ padding: 1.5rem 1rem 4rem; }}
            .page-title h1 {{ font-size: 1.35rem; }}
            .info-grid {{ grid-template-columns: 1fr; }}
            .live-badge {{ display: none; }}
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
        mapping = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEGATIVE", "LABEL_2": "POSITIVE"}
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
            f"Failed to load '{model_name}' ({model_id}). Details: {exc}"
        )


# ─────────────────────────────────────────────
# Prediction Helper
# ─────────────────────────────────────────────

def predict(text: str, model_name: str) -> tuple[str, float, float]:
    pipe    = load_pipeline(model_name)
    start   = time.time()
    result  = pipe(text)[0]
    elapsed = time.time() - start
    unified    = unify_label(result["label"], model_name)
    confidence = round(result["score"] * 100, 2)
    return unified, confidence, elapsed


# ─────────────────────────────────────────────
# Result Card
# ─────────────────────────────────────────────

def render_result_card(model_name, label, confidence, elapsed):
    css  = label.lower()
    emoji = LABEL_EMOJI[label]
    st.markdown(
        f"""
        <div class="result-card {css}">
            <div class="model-name">{model_name}</div>
            <div class="sentiment-label {css}">{emoji} {label}</div>
            <div class="confidence">{confidence:.2f}%</div>
            <div class="latency">⏱ {elapsed * 1000:.0f} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Inject styles
# ─────────────────────────────────────────────

inject_styles(st.session_state.dark_mode)


# ─────────────────────────────────────────────
# Navbar  (pure HTML — rendered once at top)
# ─────────────────────────────────────────────

theme_icon = "☀️" if st.session_state.dark_mode else "🌙"
theme_tip  = "Switch to light mode" if st.session_state.dark_mode else "Switch to dark mode"

st.markdown(
    f"""
    <div class="navbar">
        <div class="navbar-left">
            <span class="navbar-brand">Sentiment Classifier</span>
            <span class="navbar-version">v2.0</span>
        </div>
        <div class="navbar-right">
            <span class="live-badge">
                <span class="live-dot"></span>Live Inference
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Theme toggle — sits in a 0-padding column trick to float right of navbar
# We place it right after the navbar HTML so Streamlit stacks it there.
# Use a fixed-width right-aligned container via columns.
_, theme_col = st.columns([11, 1])
with theme_col:
    if st.button(theme_icon, help=theme_tip, key="theme_btn"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()


# ─────────────────────────────────────────────
# Page Body wrapper open
# ─────────────────────────────────────────────

st.markdown('<div class="page-body">', unsafe_allow_html=True)

# ── Page title ────────────────────────────────
st.markdown(
    """
    <div class="page-title">
        <h1>Sentiment Analysis</h1>
        <p>DistilBERT · RoBERTa · BERT Multilingual — Hugging Face Transformers</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────

user_text = st.text_area(
    label="Text to analyse",
    placeholder="e.g. 'The movie was absolutely wonderful — I loved every minute of it!'",
    height=110,
    key="user_text_input",
)

col_left, col_toggle, col_btn = st.columns([3, 2, 1])

with col_left:
    compare_mode = st.toggle(
        "Compare all models",
        value=False,
        help="Run all three models simultaneously.",
    )

with col_left:
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
    st.markdown("<br><br>", unsafe_allow_html=True)
    predict_clicked = st.button("Run →", type="primary", use_container_width=True)


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

    # ── Single model ──────────────────────────
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
                f"**{selected_model}** → **POSITIVE** · {confidence:.2f}% confidence · {elapsed*1000:.0f} ms"
            )
        else:
            st.error(
                f"**{selected_model}** → **NEGATIVE** · {confidence:.2f}% confidence · {elapsed*1000:.0f} ms"
            )

        with st.expander("Raw prediction details"):
            pipe = load_pipeline(selected_model)
            raw  = pipe(user_text)[0]
            st.json({
                "model"         : MODEL_CONFIG[selected_model],
                "raw_label"     : raw["label"],
                "raw_score"     : round(raw["score"], 6),
                "unified_label" : label,
                "confidence_pct": confidence,
                "inference_ms"  : round(elapsed * 1000, 1),
            })

    # ── Compare all ───────────────────────────
    else:
        st.markdown(
            '<p style="font-size:0.7rem;font-weight:600;text-transform:uppercase;'
            'letter-spacing:0.1em;color:var(--text-muted);margin-bottom:1rem;">All-Model Results</p>',
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

        if results:
            labels_list = [v[0] for v in results.values()]
            pos_count   = labels_list.count("POSITIVE")
            neg_count   = labels_list.count("NEGATIVE")
            total       = len(model_keys)

            if pos_count == total:
                icon, verdict, color = "✓", "Unanimous — POSITIVE", "var(--accent-green)"
            elif neg_count == total:
                icon, verdict, color = "✗", "Unanimous — NEGATIVE", "var(--accent-red)"
            else:
                majority = "POSITIVE" if pos_count > neg_count else "NEGATIVE"
                icon, verdict, color = "≈", f"Split — majority {majority}", "var(--accent-orange)"

            st.markdown(
                f"""
                <div class="consensus-card">
                    <div class="consensus-icon" style="color:{color};">{icon}</div>
                    <div>
                        <div class="consensus-verdict">{verdict}</div>
                        <div class="consensus-tally">{pos_count} positive · {neg_count} negative across {total} models</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Comparison table"):
                import pandas as pd
                table = pd.DataFrame([
                    {
                        "Model"        : name,
                        "Model ID"     : MODEL_CONFIG[name],
                        "Sentiment"    : f"{LABEL_EMOJI[lbl]} {lbl}",
                        "Confidence"   : f"{conf:.2f}%",
                        "Latency (ms)" : f"{elap * 1000:.0f}",
                    }
                    for name, (lbl, conf, elap) in results.items()
                ])
                st.dataframe(table, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# Info Section — Models & Label Mapping
# ─────────────────────────────────────────────

st.markdown(
    """
    <div class="info-grid">
        <div class="info-panel">
            <div class="info-panel-title">Available Models</div>
            <div class="model-row">
                <span class="model-row-name">DistilBERT (SST-2)</span>
                <span class="model-row-scope">English</span>
            </div>
            <div class="model-row">
                <span class="model-row-name">RoBERTa (Twitter)</span>
                <span class="model-row-scope">Social</span>
            </div>
            <div class="model-row">
                <span class="model-row-name">BERT Multilingual</span>
                <span class="model-row-scope">Multilingual</span>
            </div>
        </div>
        <div class="info-panel">
            <div class="info-panel-title">Label Mapping</div>
            <div class="mapping-row">
                <span class="mapping-model">DistilBERT</span>
                <span class="mapping-rule">POSITIVE / NEGATIVE</span>
            </div>
            <div class="mapping-row">
                <span class="mapping-model">RoBERTa</span>
                <span class="mapping-rule">L0,L1 → NEG &nbsp;·&nbsp; L2 → POS</span>
            </div>
            <div class="mapping-row">
                <span class="mapping-model">BERT</span>
                <span class="mapping-rule">1–3★ → NEG &nbsp;·&nbsp; 4–5★ → POS</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)  # close .page-body