"""
pages/single_scoring.py
Single-finding scorer with SHAP waterfall, gauge, flags, and human-in-the-loop override.
"""

import os
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json

from utils.model_loader import load_artefacts
from utils.inference import predict_risk, band_color, band_bg
from utils.export import to_aura_json


DOMAIN_LABELS  = {"PAM": "Privileged Access Management", "NJL": "New Joiners / Leavers",
                  "CM": "Change Management", "BR": "Backup & Restoration"}
INDUSTRIES     = ["Energy & Utilities", "Financial Services",
                  "Manufacturing", "Pharmaceuticals", "Retail"]
APP_TYPES      = ["Non-Generic", "Generic / Home-grown"]

FLAG_LABELS = {
    "flag_unauth_access":    "Unauthorised Access",
    "flag_data_loss":        "Data Loss / Integrity",
    "flag_priv_escalation":  "Privilege Escalation",
    "flag_no_logging":       "No Audit Logging",
    "flag_weak_credentials": "Weak Credentials",
}

EXAMPLE_FINDINGS = {
    "High — SAP PAM (Privilege Escalation)": {
        "observation": """During our review of privileged access controls for SAP ERP, we identified that 14 active user accounts hold both developer and production access simultaneously. This constitutes a segregation of duties (SoD) violation as users can promote unauthorised changes directly into the production environment without independent approval. No compensating monitoring controls were evidenced for these accounts during the audit period.""",
        "risk": """There is a high risk that unauthorised or erroneous changes could be made to the production SAP environment, potentially resulting in financial misstatements, data integrity issues, or regulatory non-compliance. The absence of SoD controls significantly increases the risk of fraud and undetected errors in financial processing.""",
        "domain": "PAM", "application": "SAP", "industry": "Financial Services", "app_type": "Non-Generic",
    },
    "Low — QualityTrack BR (Procedural Gap)": {
        "observation": """During our review of the backup and restoration procedures for the QualityTrack system, we observed that the backup completion logs are not formally signed off by a designated reviewer each month. Backups are being performed successfully and automated alerts confirm completion, however the sign-off step in the documented procedure is not consistently followed. No backup failures were identified during the audit period.""",
        "risk": """There is a low risk that backup completion cannot be formally evidenced in the event of an audit or regulatory review. The absence of documented sign-off may result in a minor procedural non-compliance finding. Backup integrity itself is not affected as automated monitoring confirms successful execution.""",
        "domain": "BR", "application": "QualityTrack", "industry": "Pharmaceuticals", "app_type": "Non-Generic",
    },
    "Medium — Windows AD NJL (Leaver Access)": {
        "observation": """During data analytics on Windows Active Directory, we identified 6 user accounts belonging to employees who had left the organisation between January and June 2024 that remained active beyond their termination date. The accounts were active for between 3 and 21 days post-termination before being disabled. No evidence of access activity was observed during this period for 5 of the 6 accounts; 1 account showed read-only access to shared drives on day 2 post-termination.""",
        "risk": """There is a medium risk that former employees could access sensitive company systems and data following termination, potentially leading to data exfiltration, intellectual property theft, or reputational damage. The delay in disabling accounts increases exposure during the post-termination period.""",
        "domain": "NJL", "application": "Windows Active Directory", "industry": "Manufacturing", "app_type": "Non-Generic",
    },
}


def render_gauge(score: float, band: str):
    """Render a semicircular gauge for the risk score."""
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    # Arc background segments
    zones = [(0,  40,  "#1a3a2a"),   # Low zone
             (40, 70,  "#3a2a00"),   # Medium zone
             (70, 100, "#3a0f0f")]   # High zone
    for lo, hi, col in zones:
        theta1 = 180 - lo * 1.8
        theta2 = 180 - hi * 1.8
        arc = mpatches.Wedge(center=(0, 0), r=1.0, theta1=theta2, theta2=theta1,
                              width=0.28, facecolor=col, edgecolor="none", zorder=1)
        ax.add_patch(arc)

    # Active arc (score)
    active_color = band_color(band)
    theta1_a = 180
    theta2_a = 180 - score * 1.8
    arc_a = mpatches.Wedge(center=(0, 0), r=1.0, theta1=min(theta1_a, theta2_a),
                            theta2=max(theta1_a, theta2_a),
                            width=0.28, facecolor=active_color, edgecolor="none",
                            alpha=0.9, zorder=2)
    ax.add_patch(arc_a)

    # Needle
    angle_rad = np.radians(180 - score * 1.8)
    ax.annotate("", xy=(0.72 * np.cos(angle_rad), 0.72 * np.sin(angle_rad)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=active_color,
                                lw=2.5, mutation_scale=12))

    # Center dot
    circle = plt.Circle((0, 0), 0.07, color="#0a0e1a", zorder=5)
    ax.add_patch(circle)
    circle2 = plt.Circle((0, 0), 0.05, color=active_color, zorder=6)
    ax.add_patch(circle2)

    # Score text
    ax.text(0, -0.28, f"{score:.1f}", ha="center", va="center",
            fontsize=28, fontweight="600", color=active_color,
            fontfamily="DejaVu Sans")
    ax.text(0, -0.52, band.upper(), ha="center", va="center",
            fontsize=10, fontweight="700", color=active_color,
            fontfamily="DejaVu Sans", alpha=0.9)
    ax.text(0, -0.7, "CYBER RISK SCORE", ha="center", va="center",
            fontsize=7, color="#4a5568", fontfamily="DejaVu Sans")

    # Zone labels
    for x, y, lbl, col in [
        (-1.12, 0.05, "LOW",    "#10b981"),
        (-0.1,  1.12, "MED",   "#d97706"),
        (1.0,   0.05, "HIGH",   "#e53e3e"),
    ]:
        ax.text(x, y, lbl, ha="center", va="center",
                fontsize=6.5, color=col, fontfamily="DejaVu Sans", alpha=0.7)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.85, 1.25)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def render_shap_bar(features: list, values: list, band: str):
    """Horizontal SHAP bar chart for top features."""
    if not features:
        return None

    top_n = min(15, len(features))
    feats  = features[:top_n]
    vals   = values[:top_n]

    # Clean feature names for display
    display = []
    for f in feats:
        f = f.replace("flag_", "⚑ ").replace("_", " ").replace("tfidf ", "tf·").replace("domain ", "domain·").replace("industry ", "ind·").replace("apptype ", "type·")
        display.append(f.title())

    # Reverse for bottom-to-top
    feats_r   = list(reversed(display))
    vals_r    = list(reversed(vals))
    colors    = [band_color(band) if v > 0 else "#4a5568" for v in vals_r]

    fig, ax = plt.subplots(figsize=(6, max(3, top_n * 0.38)))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    bars = ax.barh(feats_r, vals_r, color=colors, height=0.65,
                   edgecolor="none", alpha=0.85)

    # Value labels
    for bar, v in zip(bars, vals_r):
        x_pos = bar.get_width() + 0.001 if v >= 0 else bar.get_width() - 0.001
        ha    = "left" if v >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{v:+.3f}", va="center", ha=ha,
                fontsize=7.5, color="#8898aa", fontfamily="DejaVu Sans")

    ax.axvline(0, color="#2d3748", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on High-severity prediction)",
                  fontsize=8, color="#4a5568", fontfamily="DejaVu Sans")
    ax.tick_params(axis="y", labelsize=8.5, colors="#c0cce0", labelcolor="#c0cce0")
    ax.tick_params(axis="x", labelsize=7, colors="#4a5568")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#1e2d45")
    ax.set_facecolor("#111827")

    plt.tight_layout()
    return fig


# ── Ask AI helpers ─────────────────────────────────────────────────────────────

def _build_ai_system_prompt(result: dict, metadata: dict) -> str:
    """Build a rich context-aware system prompt for the AI chat."""
    shap_lines = []
    for feat, val in zip(result.get("shap_features", [])[:10], result.get("shap_values", [])[:10]):
        direction = "increases" if val > 0 else "decreases"
        clean = (feat.replace("flag_", "")
                     .replace("_", " ")
                     .replace("tfidf ", "keyword: ")
                     .replace("domain ", "domain: ")
                     .replace("industry ", "industry: ")
                     .replace("apptype ", "app type: ")
                     .title())
        shap_lines.append(f"  • {clean} ({direction} risk, SHAP={val:+.4f})")
    shap_str = "\n".join(shap_lines) if shap_lines else "  Not available"

    flags_active = [FLAG_LABELS[k] for k, v in result.get("flags", {}).items() if v]
    flags_str    = ", ".join(flags_active) if flags_active else "None detected"
    domain_full  = f"{metadata.get('control_domain')} — {DOMAIN_LABELS.get(metadata.get('control_domain', ''), '')}"

    return f"""You are an expert ITGC (IT General Controls) cyber risk analyst embedded in a scoring tool used by auditors at a professional services firm.

An XGBoost machine learning model has just scored an ITGC deficiency finding. Your role is to explain the score in clear, professional language and answer follow-up questions from the auditor.

═══════════════════════════════════════════
FINDING DETAILS
═══════════════════════════════════════════
Control Domain : {domain_full}
Application    : {metadata.get('application', 'Unknown')}
Industry       : {metadata.get('industry', 'Unknown')}
App Type       : {metadata.get('app_type', 'Unknown')}

OBSERVATION:
{metadata.get('observation', '')}

RISK STATEMENT:
{metadata.get('risk', '')}

═══════════════════════════════════════════
MODEL SCORING RESULTS
═══════════════════════════════════════════
Risk Score      : {result.get('risk_score', 0):.1f} / 100
Risk Band       : {result.get('risk_band', '').upper()}
Predicted Class : {result.get('predicted_class', '').upper()}
P(High)         : {result.get('p_high', 0)*100:.1f}%
P(Medium)       : {result.get('p_medium', 0)*100:.1f}%
P(Low)          : {result.get('p_low', 0)*100:.1f}%

Observation length         : {result.get('obs_word_count', '—')} words
High-severity keywords     : {result.get('high_sev_kw_count', '—')}
Quantity finding detected  : {'Yes' if result.get('has_quantity_finding') else 'No'}
Application tier           : {result.get('app_tier', '—')}

CYBER RISK FLAGS DETECTED:
{flags_str}

TOP SHAP FEATURE CONTRIBUTIONS:
{shap_str}

═══════════════════════════════════════════
YOUR GUIDELINES
═══════════════════════════════════════════
- Explain the score clearly and concisely in professional audit language
- Reference specific SHAP drivers and flags to justify the classification
- Keep responses focused — no unnecessary padding
- Be transparent about model uncertainty where relevant
- If asked about override decisions, provide balanced, objective guidance
- Use markdown formatting (bold key terms, bullet points where helpful)
"""


def _get_anthropic_client():
    """Get Anthropic client, checking Streamlit secrets then environment."""
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        return None


def _stream_ai_response(client, messages: list, system_prompt: str):
    """Yield text tokens from a streaming Claude response."""
    import anthropic as _anthropic
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


@st.dialog("AI Risk Analyst", width="large")
def _ai_dialog(result: dict, metadata: dict):
    """Perplexity-style modal chat window for AI score explanation."""

    # ── Dialog styles ──────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* Scrollable message area */
    .ai-messages-wrap {
        max-height: 52vh;
        overflow-y: auto;
        padding-right: 0.3rem;
        margin-bottom: 0.8rem;
    }
    /* Assistant bubble */
    .ai-bubble-assistant {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 14px 14px 14px 4px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1rem;
        font-size: 0.93rem;
        line-height: 1.75;
        color: #e2e8f0;
    }
    .ai-bubble-assistant h3 {
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.07em !important;
        text-transform: uppercase !important;
        color: #4a5568 !important;
        margin: 1rem 0 0.4rem !important;
    }
    .ai-bubble-assistant h3:first-child { margin-top: 0 !important; }
    /* User bubble */
    .ai-bubble-user {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px 14px 4px 14px;
        padding: 0.7rem 1.1rem;
        margin: 0.5rem 0 1rem 3rem;
        font-size: 0.9rem;
        color: #a5b4fc;
        line-height: 1.6;
    }
    /* Header strip */
    .ai-dialog-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #1e2d45;
    }
    .ai-dialog-title {
        font-size: 1rem;
        font-weight: 700;
        color: #f0f4ff;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .ai-model-pill {
        font-size: 0.67rem;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.08em;
        color: #6366f1;
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        padding: 0.18rem 0.55rem;
        border-radius: 20px;
    }
    /* Input row */
    .ai-input-row {
        border-top: 1px solid #1e2d45;
        padding-top: 0.9rem;
        margin-top: 0.5rem;
    }
    /* Hide default chat avatars — we style our own bubbles */
    [data-testid="stChatMessageContent"] { padding: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────────────
    band  = result.get("risk_band", "").upper()
    score = result.get("risk_score", 0)
    bc    = band_color(band.lower())

    st.markdown(f"""
    <div class="ai-dialog-header">
      <div class="ai-dialog-title">
        🤖 AI Risk Analyst
        <span style="font-size:0.78rem;font-weight:400;color:#4a5568;margin-left:0.3rem">
          — {result.get('predicted_class','').upper()} RISK &nbsp;
          <span style="color:{bc};font-weight:600">{score:.1f}/100</span>
        </span>
      </div>
      <span class="ai-model-pill">claude-opus-4-6 · anthropic</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Client check ───────────────────────────────────────────────────────
    client = _get_anthropic_client()
    if client is None:
        st.error(
            "**API key not configured.** "
            "Add `ANTHROPIC_API_KEY` in Streamlit Cloud → App Settings → Secrets.",
            icon="⚠️",
        )
        return

    system_prompt = (
        st.session_state.get("ai_system_prompt")
        or _build_ai_system_prompt(result, metadata)
    )

    # ── Initial explanation (first open) ───────────────────────────────────
    if not st.session_state.get("ai_initial_done"):
        init_prompt = (
            f"This ITGC finding was scored **{band} RISK ({score:.1f} / 100)** "
            f"by an XGBoost classifier. Explain the rationale clearly and professionally.\n\n"
            "Structure your response exactly as follows — use these exact section headings:\n\n"
            "### Verdict\n"
            "One bold sentence summarising the classification.\n\n"
            "### Why this score?\n"
            "2–3 sentences on the overall rationale.\n\n"
            "### Key Drivers\n"
            "Bullet list of the top SHAP features and what they mean in plain English.\n\n"
            "### Risk Flags\n"
            "List any detected risk flags and their significance (or note if none).\n\n"
            "### Auditor Guidance\n"
            "One sentence on whether to accept, escalate, or consider overriding the model score."
        )

        st.markdown('<div class="ai-messages-wrap">', unsafe_allow_html=True)
        with st.chat_message("assistant", avatar="🤖"):
            response_text = st.write_stream(
                _stream_ai_response(client, [{"role": "user", "content": init_prompt}], system_prompt)
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.session_state["ai_messages"] = [
            {"role": "user",      "content": init_prompt},
            {"role": "assistant", "content": response_text},
        ]
        st.session_state["ai_initial_done"] = True

    else:
        # ── Existing conversation ──────────────────────────────────────────
        st.markdown('<div class="ai-messages-wrap">', unsafe_allow_html=True)
        for msg in st.session_state["ai_messages"][1:]:   # skip hidden init prompt
            if msg["role"] == "assistant":
                st.markdown(f'<div class="ai-bubble-assistant">{_md_to_html_passthrough(msg["content"])}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-bubble-user">{msg["content"]}</div>',
                            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Follow-up input ────────────────────────────────────────────────────
    st.markdown('<div class="ai-input-row"></div>', unsafe_allow_html=True)
    col_inp, col_btn = st.columns([6, 1])
    with col_inp:
        followup = st.text_input(
            "followup",
            placeholder="Ask a follow-up question…",
            label_visibility="collapsed",
            key="ai_followup_input",
        )
    with col_btn:
        send = st.button("Send ↑", use_container_width=True, key="ai_send_btn")

    if send and followup.strip():
        st.session_state["ai_messages"].append({"role": "user", "content": followup.strip()})
        with st.chat_message("assistant", avatar="🤖"):
            response_text = st.write_stream(
                _stream_ai_response(client, st.session_state["ai_messages"], system_prompt)
            )
        st.session_state["ai_messages"].append({"role": "assistant", "content": response_text})
        st.rerun()


def _md_to_html_passthrough(text: str) -> str:
    """Pass markdown through — Streamlit's unsafe_allow_html renders it as-is,
    so we just escape nothing and rely on st.markdown for assistant bubbles
    that use the chat_message widget. For custom div bubbles we return raw text."""
    return text


def _render_ask_ai(result: dict, metadata: dict):
    """Render the Ask AI CTA and open the dialog on click."""
    st.markdown("---")

    # Session state defaults
    for key, val in [("ai_chat_open", False), ("ai_messages", []),
                     ("ai_initial_done", False), ("ai_system_prompt", "")]:
        if key not in st.session_state:
            st.session_state[key] = val

    # CTA
    st.markdown("""
    <div style="text-align:center;padding:1.6rem 0 0.6rem">
      <div style="font-size:0.75rem;color:#4a5568;letter-spacing:0.07em;
                  text-transform:uppercase;margin-bottom:0.85rem">
        Want to understand the reasoning behind this score?
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1.6, 2, 1.6])
    with col_c:
        if st.button("✦  Ask AI — Explain this Score", use_container_width=True, key="open_ai_chat"):
            st.session_state["ai_chat_open"]     = True
            st.session_state["ai_messages"]      = []
            st.session_state["ai_initial_done"]  = False
            st.session_state["ai_system_prompt"] = _build_ai_system_prompt(result, metadata)
            _ai_dialog(result, metadata)

    # Re-open dialog on page reruns while still active
    if st.session_state.get("ai_chat_open") and st.session_state.get("ai_initial_done"):
        _ai_dialog(result, metadata)


def render():
    st.markdown("## 🎯 Single Finding Scorer")
    st.markdown('<div class="section-header">Enter an ITGC deficiency finding to generate a cyber risk score with SHAP explainability</div>', unsafe_allow_html=True)

    artefacts = load_artefacts()

    # ── Example selector ───────────────────────────────────────────────────────
    with st.expander("📄 Load an example finding", expanded=False):
        ex_choice = st.selectbox("Example findings", ["— select —"] + list(EXAMPLE_FINDINGS.keys()))
        if ex_choice != "— select —":
            ex = EXAMPLE_FINDINGS[ex_choice]
            if st.button("Load this example →"):
                st.session_state["obs_input"]     = ex["observation"]
                st.session_state["risk_input"]    = ex["risk"]
                st.session_state["domain_input"]  = ex["domain"]
                st.session_state["app_input"]     = ex["application"]
                st.session_state["ind_input"]     = ex["industry"]
                st.session_state["type_input"]    = ex["app_type"]

    st.markdown("---")

    # ── Input form ─────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-header">Audit Finding Text</div>', unsafe_allow_html=True)
        observation = st.text_area(
            "Observation",
            value=st.session_state.get("obs_input", ""),
            height=160,
            placeholder='During our review of… / During data analytics on…',
            help="Full observation text from the audit finding as written by the auditor.",
            key="obs_area"
        )
        risk = st.text_area(
            "Risk Statement",
            value=st.session_state.get("risk_input", ""),
            height=120,
            placeholder="There is a [severity] risk that…",
            help="Full risk text from the audit finding.",
            key="risk_area"
        )
    with col_right:
        st.markdown('<div class="section-header">Finding Metadata</div>', unsafe_allow_html=True)

        domain_options = list(DOMAIN_LABELS.keys())
        default_domain = domain_options.index(st.session_state.get("domain_input", "PAM")) if st.session_state.get("domain_input") in domain_options else 0
        control_domain = st.selectbox(
            "Control Domain",
            domain_options,
            index=default_domain,
            format_func=lambda k: f"{k} — {DOMAIN_LABELS[k]}",
        )

        application = st.text_input(
            "Application / System",
            value=st.session_state.get("app_input", ""),
            placeholder="e.g. SAP, Windows AD, Oracle ERP",
        )

        default_ind = INDUSTRIES.index(st.session_state.get("ind_input")) if st.session_state.get("ind_input") in INDUSTRIES else 2
        industry = st.selectbox("Industry", INDUSTRIES, index=default_ind)

        default_type = APP_TYPES.index(st.session_state.get("type_input")) if st.session_state.get("type_input") in APP_TYPES else 0
        app_type = st.selectbox("Application Type", APP_TYPES, index=default_type)

        st.markdown("<br>", unsafe_allow_html=True)
        score_btn = st.button("▶  Score Finding", use_container_width=True)

    # ── Run inference ──────────────────────────────────────────────────────────
    if score_btn:
        if not observation.strip() or not risk.strip():
            st.warning("⚠️  Please enter both an Observation and a Risk statement.")
            return
        if not artefacts:
            st.error("Model artefacts not loaded. Place the `model_artefacts/` folder in the app root directory.")
            return

        with st.spinner("Scoring finding…"):
            result = predict_risk(
                observation    = observation,
                risk           = risk,
                control_domain = control_domain,
                application    = application or "Unknown",
                industry       = industry,
                app_type       = app_type,
                artefacts      = artefacts,
                compute_shap   = True,
            )
        st.session_state["last_result"]   = result
        st.session_state["last_metadata"] = {
            "observation":    observation,
            "risk":           risk,
            "control_domain": control_domain,
            "application":    application or "Unknown",
            "industry":       industry,
            "app_type":       app_type,
        }
        # Reset AI chat whenever a new score is computed
        st.session_state["ai_chat_open"]   = False
        st.session_state["ai_messages"]    = []
        st.session_state["ai_initial_done"] = False

    # ── Results panel ──────────────────────────────────────────────────────────
    if "last_result" not in st.session_state:
        return

    result   = st.session_state["last_result"]
    metadata = st.session_state["last_metadata"]
    band     = result["risk_band"]
    score    = result["risk_score"]
    bc       = band_color(band)
    bb       = band_bg(band)

    st.markdown("---")
    st.markdown("### 📊 Scoring Results")

    # Row 1 — gauge + probabilities + flags
    col_g, col_p, col_f = st.columns([2, 1.5, 2], gap="large")

    with col_g:
        st.markdown('<div class="section-header">Risk Score</div>', unsafe_allow_html=True)
        gauge_fig = render_gauge(score, band)
        st.pyplot(gauge_fig, use_container_width=True)
        plt.close()

        # ── PwC Scale conversion (1 = most critical, 5 = least) ──────────
        pwc_levels = [
            (1, "Critical",    "#e53e3e", (80, 100)),
            (2, "High",        "#f97316", (60,  79)),
            (3, "Moderate",    "#d97706", (40,  59)),
            (4, "Low",         "#3b82f6", (20,  39)),
            (5, "Negligible",  "#10b981", ( 0,  19)),
        ]
        pwc_rating = next(
            (lvl for lvl, _, _, (lo, hi) in pwc_levels if lo <= score <= hi),
            1
        )
        pwc_label  = next(lbl for lvl, lbl, _, _ in pwc_levels if lvl == pwc_rating)
        pwc_colour = next(col for lvl, _, col, _ in pwc_levels if lvl == pwc_rating)

        # Five-box strip
        boxes = ""
        for lvl, lbl, col, _ in pwc_levels:
            active = lvl == pwc_rating
            bg     = col if active else "#1a2236"
            border = col if active else "#1e2d45"
            weight = "700" if active else "400"
            opacity = "1" if active else "0.35"
            boxes += (
                f'<div style="flex:1;text-align:center;padding:0.35rem 0.1rem;'
                f'background:{bg};border:1px solid {border};border-radius:6px;'
                f'opacity:{opacity};transition:all .2s">'
                f'<div style="font-size:1rem;font-weight:{weight};color:#f0f4ff">{lvl}</div>'
                f'<div style="font-size:0.55rem;color:#8898aa;letter-spacing:0.04em;'
                f'margin-top:0.1rem">{lbl.upper()}</div>'
                f'</div>'
            )

        st.markdown(f"""
        <div style="margin-top:0.6rem">
          <div style="font-size:0.68rem;color:#4a5568;letter-spacing:0.07em;
                      text-transform:uppercase;margin-bottom:0.45rem;text-align:center">
            PwC Rating Scale
          </div>
          <div style="display:flex;gap:0.3rem">{boxes}</div>
          <div style="text-align:center;margin-top:0.5rem">
            <span style="font-size:0.72rem;color:{pwc_colour};font-weight:600;
                         font-family:'DM Mono',monospace;letter-spacing:0.05em">
              RATING {pwc_rating} — {pwc_label.upper()}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_p:
        st.markdown('<div class="section-header">Class Probabilities</div>', unsafe_allow_html=True)
        for cls, prob, col in [
            ("High",   result["p_high"],   "#e53e3e"),
            ("Medium", result["p_medium"], "#d97706"),
            ("Low",    result["p_low"],    "#10b981"),
        ]:
            pct = prob * 100
            st.markdown(f"""
            <div style="margin-bottom:0.8rem">
              <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem">
                <span style="font-size:0.8rem;color:{col};font-weight:600">{cls}</span>
                <span style="font-size:0.8rem;font-family:'DM Mono',monospace;color:{col}">{pct:.1f}%</span>
              </div>
              <div style="background:#1a2236;border-radius:4px;height:8px;overflow:hidden">
                <div style="background:{col};width:{pct}%;height:100%;border-radius:4px;opacity:0.85;transition:width 0.5s ease"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:0.8rem;padding:0.5rem 0.8rem;background:{bb};border:1px solid {bc};border-radius:8px;text-align:center">
          <div style="font-size:0.7rem;color:{bc};font-family:'DM Mono',monospace;letter-spacing:0.08em">PREDICTED CLASS</div>
          <div style="font-size:1.1rem;font-weight:700;color:{bc}">{result['predicted_class'].upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_f:
        st.markdown('<div class="section-header">Cyber Risk Flags</div>', unsafe_allow_html=True)
        flags = result.get("flags", {})
        for flag_key, flag_label in FLAG_LABELS.items():
            active = flags.get(flag_key, 0)
            if active:
                st.markdown(f'<span class="flag-chip">⚑ {flag_label}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="flag-chip flag-chip-off">· {flag_label}</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Quick stats
        st.markdown('<div class="section-header">Signal Metrics</div>', unsafe_allow_html=True)
        for label, val in [
            ("Observation length", f"{result.get('obs_word_count', '—')} words"),
            ("High-severity keywords", str(result.get("high_sev_kw_count", "—"))),
            ("Quantity finding detected", "Yes" if result.get("has_quantity_finding") else "No"),
            ("Application tier", str(result.get("app_tier", "—"))),
        ]:
            st.markdown(f"""
            <div class="info-row">
              <span class="info-label">{label}</span>
              <span class="info-value">{val}</span>
            </div>""", unsafe_allow_html=True)

    # ── SHAP Explainability ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 SHAP Explainability")
    st.markdown('<div class="section-header">Feature contributions to the High-severity class prediction (positive = increases risk score)</div>', unsafe_allow_html=True)

    shap_feats = result.get("shap_features", [])
    shap_vals  = result.get("shap_values", [])

    if shap_feats:
        shap_fig = render_shap_bar(shap_feats, shap_vals, band)
        if shap_fig:
            st.pyplot(shap_fig, use_container_width=True)
            plt.close()

        # Top drivers in plain English
        with st.expander("📝 Top feature drivers (plain English)", expanded=False):
            for feat, val in zip(shap_feats[:8], shap_vals[:8]):
                direction = "↑ increases" if val > 0 else "↓ decreases"
                clean = feat.replace("flag_", "").replace("_", " ").replace("tfidf ", "TF-IDF token: ").title()
                col_txt = "#fca5a5" if val > 0 else "#86efac"
                st.markdown(f'<div style="padding:0.3rem 0;font-size:0.85rem"><span style="color:{col_txt};font-weight:600">{direction} severity</span> · <span style="color:#c0cce0">{clean}</span> <span style="color:#4a5568;font-family:\'DM Mono\',monospace">({val:+.4f})</span></div>', unsafe_allow_html=True)
    elif "shap_error" in result:
        st.warning(f"SHAP computation skipped: {result['shap_error']}")
    else:
        st.info("SHAP values not available for this result.")

    # ── Ask AI ─────────────────────────────────────────────────────────────────
    _render_ask_ai(result, metadata)

    # ── Human-in-the-loop override ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 👤 Auditor Review")

    with st.container():
        st.markdown('<div class="override-box">', unsafe_allow_html=True)

        col_ov1, col_ov2 = st.columns([1, 2], gap="large")

        with col_ov1:
            override_on = st.checkbox("Apply auditor override", value=False,
                                       help="Tick to override the model's risk band with your professional judgement.")
            if override_on:
                override_band = st.selectbox("Override Risk Band", ["High", "Medium", "Low"],
                                              index=["High", "Medium", "Low"].index(band))
                override_by = st.text_input("Auditor name / ID", placeholder="e.g. JJ · Senior Auditor")
            else:
                override_band = None
                override_by   = None

        with col_ov2:
            if override_on:
                override_reason = st.text_area(
                    "Override rationale",
                    height=100,
                    placeholder="Document your professional judgement for overriding the model score…",
                )
            else:
                override_reason = None
                st.markdown(f"""
                <div style="padding:1rem;border:1px solid {bc};border-radius:10px;background:{bb}">
                  <div style="font-size:0.72rem;font-family:'DM Mono',monospace;color:{bc};letter-spacing:0.06em;margin-bottom:0.4rem">MODEL RECOMMENDATION</div>
                  <div style="font-size:1.5rem;font-weight:700;color:{bc}">{band.upper()} RISK &nbsp; <span style="font-size:1rem;opacity:0.7">({score:.1f}/100)</span></div>
                  <div style="font-size:0.8rem;color:#8898aa;margin-top:0.3rem">Accept this classification to proceed to export.</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Export ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📤 Export")

    final_band   = override_band if override_on and override_band else band
    final_reason = override_reason if override_on else None

    export_meta = {
        **metadata,
        "auditor_override": override_on,
        "override_band":    final_band,
        "override_reason":  final_reason,
        "override_by":      override_by if override_on else None,
    }

    aura_json = to_aura_json(result, export_meta)

    col_e1, col_e2 = st.columns(2, gap="medium")
    with col_e1:
        st.download_button(
            label="⬇  Download Aura JSON",
            data=aura_json,
            file_name=f"itgc_risk_{metadata['control_domain']}_{score:.0f}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_e2:
        with st.expander("👁  Preview JSON payload"):
            st.code(aura_json, language="json")
