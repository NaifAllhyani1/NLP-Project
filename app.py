"""
app.py  –  Sentiment Analysis Dashboard
Run with:  streamlit run app.py
"""

import time
import streamlit as st
from transformers import pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_styles(dark: bool):
    if dark:
        p = dict(
            bg_base="#0d1117", bg_surface="#161b22", bg_card="#1c2330",
            bg_input="#0d1117", border="#30363d", border_sub="#21262d",
            text_p="#e6edf3", text_s="#8b949e", text_m="#484f58",
            green="#3fb950", red="#f85149", blue="#58a6ff", orange="#d29922",
            glow_g="rgba(63,185,80,.12)", glow_r="rgba(248,81,73,.12)",
            shadow="0 8px 32px rgba(0,0,0,.45)", shadow_s="0 2px 8px rgba(0,0,0,.3)",
            nav="#161b22", toggle_bg="#21262d", toggle_knob="#e6edf3",
        )
    else:
        p = dict(
            bg_base="#f6f8fa", bg_surface="#ffffff", bg_card="#ffffff",
            bg_input="#f0f3f7", border="#d0d7de", border_sub="#eaeef2",
            text_p="#1c2128", text_s="#57606a", text_m="#8c959f",
            green="#1a7f37", red="#cf222e", blue="#0969da", orange="#9a6700",
            glow_g="rgba(26,127,55,.07)", glow_r="rgba(207,34,46,.07)",
            shadow="0 8px 32px rgba(0,0,0,.08)", shadow_s="0 2px 8px rgba(0,0,0,.05)",
            nav="#ffffff", toggle_bg="#d0d7de", toggle_knob="#1c2128",
        )

    sun = "#f5a623"
    moon = "#8b949e"
    knob_icon = "☀️" if dark else "🌙"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Geist:wght@300;400;500;600&display=swap');

:root {{
  --bg-base:       {p['bg_base']};
  --bg-surface:    {p['bg_surface']};
  --bg-card:       {p['bg_card']};
  --bg-input:      {p['bg_input']};
  --border:        {p['border']};
  --border-sub:    {p['border_sub']};
  --text-p:        {p['text_p']};
  --text-s:        {p['text_s']};
  --text-m:        {p['text_m']};
  --green:         {p['green']};
  --red:           {p['red']};
  --blue:          {p['blue']};
  --orange:        {p['orange']};
  --glow-g:        {p['glow_g']};
  --glow-r:        {p['glow_r']};
  --shadow:        {p['shadow']};
  --shadow-s:      {p['shadow_s']};
  --nav:           {p['nav']};
}}

/* ── Reset ── */
html, body, [class*="css"], .stApp, .block-container {{
  font-family: 'Geist', sans-serif !important;
  background-color: var(--bg-base) !important;
  color: var(--text-p) !important;
}}
#MainMenu, footer, header[data-testid="stHeader"] {{ display:none !important; }}
[data-testid="collapsedControl"], section[data-testid="stSidebar"] {{ display:none !important; }}

.block-container {{
  padding: 0 !important;
  max-width: 100% !important;
}}

/* ── Navbar ── */
.navbar {{
  position: sticky; top: 0; z-index: 1000;
  display: flex; align-items: center; justify-content: space-between;
  height: 58px; padding: 0 2.5rem;
  background: var(--nav);
  border-bottom: 1px solid var(--border);
}}
.nav-brand {{
  display: flex; align-items: center; gap: 0.6rem;
}}
.nav-brand-name {{
  font-size: 0.92rem; font-weight: 600;
  color: var(--text-p); letter-spacing: -0.01em;
}}
.nav-version {{
  font-family: 'Geist Mono', monospace;
  font-size: 0.62rem; color: var(--text-m);
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 4px; padding: 1px 6px;
}}
.nav-right {{
  display: flex; align-items: center; gap: 1rem;
}}
.live-pill {{
  display: inline-flex; align-items: center; gap: 0.4rem;
  font-family: 'Geist Mono', monospace; font-size: 0.67rem;
  color: var(--text-s);
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 20px; padding: 4px 12px;
}}
.live-dot {{
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--green);
  animation: blink 2s ease-in-out infinite;
}}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:.3}} }}

/* ── Theme toggle pill ── */
.theme-pill {{
  display: flex; align-items: center; gap: 0.5rem;
  background: {p['toggle_bg']}; border: 1px solid var(--border);
  border-radius: 20px; padding: 4px 4px 4px 12px;
  cursor: pointer; user-select: none;
  font-size: 0.72rem; font-weight: 500; color: var(--text-s);
  transition: background 0.2s, border-color 0.2s;
}}
.theme-pill:hover {{ border-color: var(--text-s); }}
.theme-knob {{
  width: 26px; height: 26px; border-radius: 50%;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.75rem;
  box-shadow: 0 1px 4px rgba(0,0,0,.25);
  transition: transform 0.2s;
}}

/* ── Page wrapper ── */
.page-wrap {{
  max-width: 780px; margin: 0 auto;
  padding: 2.5rem 1.5rem 5rem;
}}

/* ── Page title ── */
.page-title {{ margin-bottom: 2rem; }}
.page-title h1 {{
  font-size: 1.8rem; font-weight: 600;
  letter-spacing: -0.03em; color: var(--text-p);
  margin: 0 0 0.3rem;
}}
.page-title p {{
  font-family: 'Geist Mono', monospace;
  font-size: 0.73rem; color: var(--text-m); margin: 0;
}}

/* ── Input card ── */
.input-card {{
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 12px; padding: 1.75rem;
  margin-bottom: 1.25rem;
  box-shadow: var(--shadow-s);
}}

/* ── Textarea ── */
.stTextArea textarea {{
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-p) !important;
  font-family: 'Geist', sans-serif !important;
  font-size: 0.9rem !important; resize: vertical !important;
  transition: border-color .2s !important;
}}
.stTextArea textarea:focus {{
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(88,166,255,.1) !important;
  outline: none !important;
}}
.stTextArea label, .stSelectbox label {{
  color: var(--text-s) !important; font-size: 0.73rem !important;
  font-weight: 600 !important; text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}}

/* ── Selectbox ── */
.stSelectbox > div > div {{
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important; color: var(--text-p) !important;
}}

/* ── Toggle ── */
.stToggle label {{ color: var(--text-s) !important; font-size: 0.85rem !important; }}

/* ── Run button ── */
.stButton > button[kind="primary"] {{
  background: var(--text-p) !important;
  color: var(--bg-base) !important;
  border: none !important; border-radius: 8px !important;
  font-family: 'Geist', sans-serif !important;
  font-weight: 600 !important; font-size: 0.87rem !important;
  letter-spacing: 0.02em !important; height: 42px !important;
  transition: opacity .15s, transform .1s !important;
  box-shadow: var(--shadow-s) !important;
}}
.stButton > button[kind="primary"]:hover {{
  opacity: 0.85 !important; transform: translateY(-1px) !important;
}}
.stButton > button[kind="primary"]:active {{
  transform: translateY(0px) !important;
}}

/* ── Secondary buttons (theme toggle fallback) ── */
.stButton > button:not([kind="primary"]) {{
  background: transparent !important; color: var(--text-p) !important;
  border: 1px solid var(--border) !important; border-radius: 20px !important;
  font-size: 0.8rem !important; padding: 4px 14px !important;
  transition: border-color .15s !important;
}}
.stButton > button:not([kind="primary"]):hover {{
  border-color: var(--text-s) !important;
}}

/* ── Metrics ── */
[data-testid="metric-container"] {{
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important; padding: 1.1rem 1.4rem !important;
}}
[data-testid="metric-container"] label {{
  color: var(--text-m) !important; font-size: 0.68rem !important;
  font-weight: 600 !important; text-transform: uppercase !important;
  letter-spacing: 0.09em !important;
}}
[data-testid="stMetricValue"] {{
  color: var(--text-p) !important; font-size: 1.1rem !important;
  font-weight: 600 !important;
}}

/* ── Result cards ── */
.result-card {{
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.75rem 1.25rem; text-align: center;
  box-shadow: var(--shadow-s); transition: box-shadow .2s, transform .2s;
}}
.result-card:hover {{ box-shadow: var(--shadow); transform: translateY(-2px); }}
.result-card.positive {{
  border-color: var(--green);
  background: linear-gradient(160deg, var(--glow-g), var(--bg-card) 55%);
}}
.result-card.negative {{
  border-color: var(--red);
  background: linear-gradient(160deg, var(--glow-r), var(--bg-card) 55%);
}}
.result-card .model-name {{
  font-family: 'Geist Mono', monospace; font-size: 0.63rem;
  color: var(--text-m); letter-spacing: .12em; text-transform: uppercase;
  margin-bottom: 1rem;
}}
.result-card .sentiment-label {{
  font-size: 1.4rem; font-weight: 600; margin-bottom: .45rem;
}}
.result-card .sentiment-label.positive {{ color: var(--green); }}
.result-card .sentiment-label.negative {{ color: var(--red); }}
.result-card .confidence {{
  font-family: 'Geist Mono', monospace; font-size: .88rem; color: var(--text-s);
  margin-bottom: .3rem;
}}
.result-card .latency {{
  font-family: 'Geist Mono', monospace; font-size: .63rem; color: var(--text-m);
}}

/* ── Alerts ── */
.stAlert {{ border-radius: 8px !important; }}
.stSuccess {{ background: var(--glow-g) !important; border-color: var(--green) !important; }}
.stError   {{ background: var(--glow-r) !important; border-color: var(--red) !important; }}
.stInfo    {{ background: rgba(88,166,255,.07) !important; border-color: var(--blue) !important; border-radius:8px !important; }}

/* ── Expander ── */
.streamlit-expanderHeader {{
  background: var(--bg-card) !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important; color: var(--text-s) !important; font-size: .8rem !important;
}}
.streamlit-expanderContent {{
  background: var(--bg-card) !important; border: 1px solid var(--border) !important;
  border-top: none !important; border-radius: 0 0 8px 8px !important;
}}

/* ── Consensus card ── */
.consensus-card {{
  display: flex; align-items: center; gap: 1rem;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.25rem 1.5rem; margin-top: 1.25rem;
}}
.consensus-icon {{ font-size: 1.5rem; flex-shrink: 0; }}
.consensus-verdict {{ font-size: 1rem; font-weight: 600; color: var(--text-p); }}
.consensus-tally {{
  font-family: 'Geist Mono', monospace; font-size: .7rem;
  color: var(--text-m); margin-top: .2rem;
}}

/* ── Info grid ── */
.info-grid {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 2.5rem;
}}
.info-panel {{
  background: var(--bg-surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.25rem 1.5rem;
}}
.info-panel-title {{
  font-size: .67rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: .1em; color: var(--text-m); margin-bottom: 1rem;
}}
.model-row, .mapping-row {{
  display: flex; align-items: center; justify-content: space-between;
  padding: .5rem 0; border-bottom: 1px solid var(--border-sub);
}}
.model-row:last-child, .mapping-row:last-child {{ border-bottom: none; }}
.model-row-name, .mapping-model {{
  font-size: .82rem; font-weight: 500; color: var(--text-p);
}}
.model-row-scope {{
  font-family: 'Geist Mono', monospace; font-size: .65rem; color: var(--text-m);
  background: var(--bg-input); border: 1px solid var(--border);
  border-radius: 4px; padding: 2px 7px;
}}
.mapping-rule {{
  font-family: 'Geist Mono', monospace; font-size: .67rem; color: var(--text-s);
  text-align: right;
}}

/* ── Divider ── */
hr {{ border:none !important; border-top:1px solid var(--border-sub) !important; margin:1.25rem 0 !important; }}

/* ── Responsive ── */
@media (max-width:680px) {{
  .navbar {{ padding: 0 1rem; }}
  .page-wrap {{ padding: 1.5rem 1rem 4rem; }}
  .page-title h1 {{ font-size: 1.4rem; }}
  .info-grid {{ grid-template-columns: 1fr; }}
  .live-pill {{ display:none; }}
}}
</style>""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
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


# ── Label unification ─────────────────────────────────────────────────────────
def unify_label(raw_label: str, model_name: str) -> str:
    label = raw_label.strip().upper()
    if model_name == "DistilBERT":
        return label
    elif model_name == "RoBERTa":
        m = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEGATIVE", "LABEL_2": "POSITIVE"}
        if label not in m: raise ValueError(f"Unexpected RoBERTa label: '{raw_label}'")
        return m[label]
    elif model_name == "BERT":
        d = label.split()[0]
        if not d.isdigit(): raise ValueError(f"Unexpected BERT label: '{raw_label}'")
        return "POSITIVE" if int(d) >= 4 else "NEGATIVE"
    raise ValueError(f"Unknown model: '{model_name}'")


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(model_name: str):
    try:
        return pipeline("text-classification", model=MODEL_CONFIG[model_name],
                        truncation=True, max_length=512)
    except Exception as e:
        raise RuntimeError(f"Failed to load '{model_name}'. Details: {e}")


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(text: str, model_name: str) -> tuple[str, float, float]:
    pipe  = load_pipeline(model_name)
    t0    = time.time()
    res   = pipe(text)[0]
    return unify_label(res["label"], model_name), round(res["score"]*100, 2), time.time()-t0


# ── Result card ───────────────────────────────────────────────────────────────
def render_result_card(model_name, label, confidence, elapsed):
    css = label.lower()
    st.markdown(f"""
    <div class="result-card {css}">
        <div class="model-name">{model_name}</div>
        <div class="sentiment-label {css}">{LABEL_EMOJI[label]} {label}</div>
        <div class="confidence">{confidence:.2f}%</div>
        <div class="latency">⏱ {elapsed*1000:.0f} ms</div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  RENDER
# ════════════════════════════════════════════════════════════════════

inject_styles(st.session_state.dark_mode)

# ── Navbar ────────────────────────────────────────────────────────────────────
theme_label = "Dark" if st.session_state.dark_mode else "Light"
theme_icon  = "☀️"   if st.session_state.dark_mode else "🌙"

st.markdown(f"""
<div class="navbar">
  <div class="nav-brand">
    <span class="nav-brand-name">Sentiment Classifier</span>
    <span class="nav-version">v2.0</span>
  </div>
  <div class="nav-right">
    <span class="live-pill"><span class="live-dot"></span>Live Inference</span>
  </div>
</div>""", unsafe_allow_html=True)

# Theme toggle — hoisted into navbar via CSS negative margin
_spacer, _theme_col = st.columns([9, 1])
with _theme_col:
    if st.button(f"{theme_icon} {theme_label}", key="theme_btn"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown("""
<style>
/* Hoist the theme button row up into the sticky navbar */
div[data-testid="stHorizontalBlock"]:nth-of-type(1) {
  margin-top: -52px !important;
  position: relative !important;
  z-index: 1001 !important;
  pointer-events: none;
}
div[data-testid="stHorizontalBlock"]:nth-of-type(1) * {
  pointer-events: auto;
}
div[data-testid="stHorizontalBlock"]:nth-of-type(1) [data-testid="column"]:last-child {
  display: flex !important;
  justify-content: flex-end !important;
  padding-right: 2rem !important;
  padding-top: 10px !important;
}
/* Make the secondary button look like a nav pill */
div[data-testid="stHorizontalBlock"]:nth-of-type(1) button {
  border-radius: 20px !important;
  font-size: 0.75rem !important;
  padding: 5px 14px !important;
  font-weight: 500 !important;
  height: auto !important;
  line-height: 1.4 !important;
}
</style>""", unsafe_allow_html=True)


# ── Page body ─────────────────────────────────────────────────────────────────
st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

st.markdown("""
<div class="page-title">
  <h1>Sentiment Analysis</h1>
  <p>DistilBERT · RoBERTa · BERT Multilingual — Hugging Face Transformers</p>
</div>""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
user_text = st.text_area(
    "Text to analyse",
    placeholder="e.g. 'The movie was absolutely wonderful — I loved every minute of it!'",
    height=110, key="user_text_input",
)

compare_mode = st.toggle("Compare all models", value=False,
    help="Run all three models simultaneously and compare results.")

if not compare_mode:
    sel_col, run_col = st.columns([4, 1])
    with sel_col:
        selected_display = st.selectbox("Model", list(MODEL_OPTIONS.keys()), key="model_selector")
        selected_model   = MODEL_OPTIONS[selected_display]
    with run_col:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("Run →", type="primary", use_container_width=True)
else:
    st.info("Compare mode — all three models will run on submit.")
    predict_clicked = st.button("Run All Models →", type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_clicked:
    if not user_text or not user_text.strip():
        st.warning("Please enter some text first."); st.stop()
    if len(user_text.strip()) < 3:
        st.warning("Input is too short."); st.stop()

    st.markdown("---")

    if not compare_mode:
        with st.spinner(f"Running {selected_model}…"):
            try:   label, confidence, elapsed = predict(user_text, selected_model)
            except RuntimeError as e: st.error(str(e)); st.stop()

        m1, m2, m3 = st.columns(3)
        m1.metric("Model",      selected_model)
        m2.metric("Sentiment",  f"{LABEL_EMOJI[label]} {label}")
        m3.metric("Confidence", f"{confidence:.2f}%")
        st.markdown("<br>", unsafe_allow_html=True)

        if label == "POSITIVE":
            st.success(f"**{selected_model}** → **POSITIVE** · {confidence:.2f}% confidence · {elapsed*1000:.0f} ms")
        else:
            st.error(f"**{selected_model}** → **NEGATIVE** · {confidence:.2f}% confidence · {elapsed*1000:.0f} ms")

        with st.expander("Raw prediction details"):
            pipe = load_pipeline(selected_model)
            raw  = pipe(user_text)[0]
            st.json({"model": MODEL_CONFIG[selected_model], "raw_label": raw["label"],
                     "raw_score": round(raw["score"],6), "unified_label": label,
                     "confidence_pct": confidence, "inference_ms": round(elapsed*1000,1)})

    else:
        model_keys = list(MODEL_CONFIG.keys())
        cols       = st.columns(3, gap="medium")
        results    = {}
        for col, mn in zip(cols, model_keys):
            with col:
                with st.spinner(f"{mn}…"):
                    try:
                        lbl, conf, elap = predict(user_text, mn)
                        results[mn] = (lbl, conf, elap)
                        render_result_card(mn, lbl, conf, elap)
                    except RuntimeError as e:
                        st.error(f"{mn}: {e}")

        if results:
            llist = [v[0] for v in results.values()]
            pos, neg, total = llist.count("POSITIVE"), llist.count("NEGATIVE"), len(model_keys)
            if pos == total:   icon,v,c = "✓", "Unanimous — POSITIVE", "var(--green)"
            elif neg == total: icon,v,c = "✗", "Unanimous — NEGATIVE", "var(--red)"
            else:
                maj = "POSITIVE" if pos>neg else "NEGATIVE"
                icon,v,c = "≈", f"Split — majority {maj}", "var(--orange)"
            st.markdown(f"""
            <div class="consensus-card">
              <div class="consensus-icon" style="color:{c};">{icon}</div>
              <div>
                <div class="consensus-verdict">{v}</div>
                <div class="consensus-tally">{pos} positive · {neg} negative across {total} models</div>
              </div>
            </div>""", unsafe_allow_html=True)

            with st.expander("Comparison table"):
                import pandas as pd
                st.dataframe(pd.DataFrame([{
                    "Model": n, "Model ID": MODEL_CONFIG[n],
                    "Sentiment": f"{LABEL_EMOJI[l]} {l}",
                    "Confidence": f"{c:.2f}%", "Latency (ms)": f"{e*1000:.0f}",
                } for n,(l,c,e) in results.items()]), use_container_width=True, hide_index=True)

# ── Info section ──────────────────────────────────────────────────────────────
st.markdown("""
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
</div>""", unsafe_allow_html=True)  # closes page-wrap