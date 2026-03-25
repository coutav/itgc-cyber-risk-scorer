"""
ITGC Cyber Risk Scoring Model — Streamlit Dashboard
Master Thesis · Ankit Vats (s242576) · DTU × PwC Denmark
"""

import streamlit as st

st.set_page_config(
    page_title="ITGC Cyber Risk Scorer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;500;600;700&display=swap');

:root {
    --bg-base:       #0a0e1a;
    --bg-card:       #111827;
    --bg-card2:      #1a2236;
    --border:        #1e2d45;
    --accent-red:    #e53e3e;
    --accent-amber:  #d97706;
    --accent-blue:   #3b82f6;
    --accent-green:  #10b981;
    --accent-teal:   #06b6d4;
    --text-primary:  #f0f4ff;
    --text-muted:    #8898aa;
    --text-dim:      #4a5568;
    --font-main:     'Sora', sans-serif;
    --font-mono:     'DM Mono', monospace;
}

/* Global reset */
html, body, [class*="css"] {
    font-family: var(--font-main) !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

/* Hide default Streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Main content area */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
}

/* Buttons */
div.stButton > button {
    background: var(--accent-blue) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-main) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}
div.stButton > button:hover {
    background: #2563eb !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.4) !important;
}

/* Inputs */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stSelectbox"] div {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: var(--font-main) !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-teal) !important;
    border-bottom-color: var(--accent-teal) !important;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Expander */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}
details summary {
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* Dividers */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* Custom card utility */
.pwc-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.score-gauge-label {
    font-family: var(--font-mono);
    font-size: 4.5rem;
    font-weight: 500;
    letter-spacing: -0.02em;
}

.band-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.flag-chip {
    display: inline-block;
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: var(--font-mono);
    margin: 0.15rem;
}
.flag-chip-off {
    background: rgba(74,85,104,0.2);
    border: 1px solid var(--border);
    color: var(--text-dim);
}

.sidebar-logo {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--accent-teal);
}
.sidebar-sub {
    font-size: 0.7rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
    margin-top: -0.2rem;
}

.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
    font-family: var(--font-mono);
}

.info-row {
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.info-label { color: var(--text-muted); }
.info-value { color: var(--text-primary); font-family: var(--font-mono); }

.override-box {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 12px;
    padding: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────────────────────
from pages import single_scoring, batch_scoring, model_insights, about

PAGES = {
    "🎯  Single Finding Scorer":  single_scoring,
    "📋  Batch Scoring":           batch_scoring,
    "📊  Model Insights":          model_insights,
    "ℹ️   About":                   about,
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛡️ ITGC Risk Scorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">DTU × PwC Denmark · s242576</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
    selection = st.radio("", list(PAGES.keys()), label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-header">Model Status</div>', unsafe_allow_html=True)

    # Model load status
    from utils.model_loader import load_artefacts
    artefacts = load_artefacts()
    if artefacts:
        st.success("✓ Model artefacts loaded")
        st.markdown(f'<div style="font-size:0.72rem;color:#8898aa;font-family:\'DM Mono\',monospace">XGBoost · 3-class · v1.0</div>', unsafe_allow_html=True)
    else:
        st.error("✗ Artefacts not found")
        st.markdown(
            '<div style="font-size:0.72rem;color:#8898aa">Place <code>model_artefacts/</code> folder in app root.</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#4a5568;font-family:\'DM Mono\',monospace">Control Domains</div>', unsafe_allow_html=True)
    for domain, label in [("PAM","Privileged Access Mgmt"), ("NJL","New Joiners/Leavers"), ("CM","Change Management"), ("BR","Backup & Restoration")]:
        st.markdown(f'<div style="font-size:0.72rem;padding:0.15rem 0;color:#8898aa"><span style="color:#06b6d4;font-family:\'DM Mono\',monospace">{domain}</span> · {label}</div>', unsafe_allow_html=True)

# ── Render selected page ──────────────────────────────────────────────────────
PAGES[selection].render()
