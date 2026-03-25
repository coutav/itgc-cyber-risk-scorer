"""
pages/about.py
About page — thesis context, architecture, setup guide, acknowledgements.
"""

import streamlit as st


def render():
    st.markdown("## ℹ️ About This Tool")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("""
        <div class="pwc-card">
        <div class="section-header">Project Overview</div>
        <div style="font-size:0.9rem;line-height:1.7;color:#c0cce0">
        The <b style="color:#06b6d4">ITGC Cyber Risk Scoring Model</b> is the core deliverable of a Master's Thesis
        at the <b style="color:#f0f4ff">Technical University of Denmark (DTU)</b>, conducted in collaboration with
        <b style="color:#f0f4ff">PwC Denmark's Digital Assurance, Technology Risk & Information Security</b> practice.
        </div>
        <br>
        <div style="font-size:0.88rem;line-height:1.7;color:#8898aa">
        The tool predicts the severity of IT General Controls (ITGC) audit deficiency findings,
        producing a continuous 0–100 cyber risk score with SHAP-based explainability. It addresses
        a documented gap in the audit literature: the absence of an empirically-validated, ML-based
        severity scoring system specifically designed for ITGC deficiencies.
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="pwc-card" style="margin-top:1rem">
        <div class="section-header">Technical Architecture</div>
        <div style="display:grid;grid-template-columns:1fr;gap:0.5rem;font-size:0.82rem;font-family:'DM Mono',monospace">
          <div style="display:flex;align-items:center;gap:0.8rem;padding:0.6rem;background:#1a2236;border-radius:8px;border:1px solid #1e2d45">
            <span style="color:#06b6d4;font-size:1.1rem">①</span>
            <div><span style="color:#f0f4ff">Input</span> <span style="color:#4a5568">— Observation + Risk text + metadata</span></div>
          </div>
          <div style="text-align:center;color:#4a5568">↓</div>
          <div style="display:flex;align-items:center;gap:0.8rem;padding:0.6rem;background:#1a2236;border-radius:8px;border:1px solid #1e2d45">
            <span style="color:#06b6d4;font-size:1.1rem">②</span>
            <div><span style="color:#f0f4ff">Feature Engineering</span> <span style="color:#4a5568">— 23 structured + 100 TF-IDF = 123 features</span></div>
          </div>
          <div style="text-align:center;color:#4a5568">↓</div>
          <div style="display:flex;align-items:center;gap:0.8rem;padding:0.6rem;background:#1a2236;border-radius:8px;border:1px solid #1e2d45">
            <span style="color:#06b6d4;font-size:1.1rem">③</span>
            <div><span style="color:#f0f4ff">StandardScaler</span> <span style="color:#4a5568">— Normalise to training distribution</span></div>
          </div>
          <div style="text-align:center;color:#4a5568">↓</div>
          <div style="display:flex;align-items:center;gap:0.8rem;padding:0.6rem;background:#0f2030;border-radius:8px;border:1px solid #06b6d4">
            <span style="color:#06b6d4;font-size:1.1rem">④</span>
            <div><span style="color:#06b6d4;font-weight:600">XGBoost Classifier</span> <span style="color:#4a5568">— 3-class (Low / Medium / High)</span></div>
          </div>
          <div style="text-align:center;color:#4a5568">↓</div>
          <div style="display:flex;align-items:center;gap:0.8rem;padding:0.6rem;background:#1a2236;border-radius:8px;border:1px solid #1e2d45">
            <span style="color:#06b6d4;font-size:1.1rem">⑤</span>
            <div><span style="color:#f0f4ff">Risk Score</span> <span style="color:#4a5568">— P(Low)×20 + P(Med)×55 + P(High)×90</span></div>
          </div>
          <div style="text-align:center;color:#4a5568">↓</div>
          <div style="display:flex;align-items:center;gap:0.8rem;padding:0.6rem;background:#1a2236;border-radius:8px;border:1px solid #1e2d45">
            <span style="color:#06b6d4;font-size:1.1rem">⑥</span>
            <div><span style="color:#f0f4ff">SHAP TreeExplainer</span> <span style="color:#4a5568">— Per-feature contribution to High class</span></div>
          </div>
          <div style="text-align:center;color:#4a5568">↓</div>
          <div style="display:flex;align-items:center;gap:0.8rem;padding:0.6rem;background:#1a2236;border-radius:8px;border:1px solid #1e2d45">
            <span style="color:#06b6d4;font-size:1.1rem">⑦</span>
            <div><span style="color:#f0f4ff">Human-in-the-Loop</span> <span style="color:#4a5568">— Auditor accept / override interface</span></div>
          </div>
          <div style="text-align:center;color:#4a5568">↓</div>
          <div style="display:flex;align-items:center;gap:0.8rem;padding:0.6rem;background:#1a2236;border-radius:8px;border:1px solid #1e2d45">
            <span style="color:#06b6d4;font-size:1.1rem">⑧</span>
            <div><span style="color:#f0f4ff">Aura JSON Export</span> <span style="color:#4a5568">— Structured output for downstream systems</span></div>
          </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="pwc-card">
        <div class="section-header">Thesis Info</div>
        """, unsafe_allow_html=True)

        for label, val in [
            ("Author",     "Ankit Vats"),
            ("Student ID", "s242576"),
            ("Institution","DTU"),
            ("Partner",    "PwC Denmark"),
            ("Programme",  "MSc Engineering"),
            ("Period",     "Jan – Aug 2026"),
            ("Supervisor", "DTU + PwC"),
        ]:
            st.markdown(f'<div class="info-row"><span class="info-label">{label}</span><span class="info-value">{val}</span></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="pwc-card" style="margin-top:1rem">
        <div class="section-header">ITGC Domains in Scope</div>
        """, unsafe_allow_html=True)

        for domain, label, color in [
            ("PAM", "Privileged Access Management", "#e53e3e"),
            ("NJL", "New Joiners / Leavers",         "#d97706"),
            ("CM",  "Change Management",              "#3b82f6"),
            ("BR",  "Backup & Restoration",           "#10b981"),
        ]:
            st.markdown(f"""
            <div style="padding:0.5rem 0;border-bottom:1px solid #1e2d45">
              <span style="color:{color};font-family:'DM Mono',monospace;font-weight:600;font-size:0.85rem">{domain}</span>
              <span style="color:#8898aa;font-size:0.78rem;margin-left:0.5rem">{label}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="pwc-card" style="margin-top:1rem">
        <div class="section-header">Model Stats</div>
        """, unsafe_allow_html=True)

        for label, val in [
            ("Algorithm",    "XGBoost"),
            ("Classes",      "Low / Medium / High"),
            ("Features",     "123"),
            ("Training rows","977"),
            ("CV Macro-F1",  "0.667 ± 0.032"),
            ("Score range",  "0 – 100"),
            ("XAI method",   "SHAP TreeExplainer"),
        ]:
            st.markdown(f'<div class="info-row"><span class="info-label">{label}</span><span class="info-value">{val}</span></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Setup guide ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 Setup & Deployment")

    tab1, tab2, tab3 = st.tabs(["Local Setup", "Docker", "Azure"])

    with tab1:
        st.markdown("""
        **Prerequisites:** Python 3.11+, pip

        ```bash
        # 1. Clone / unzip the app directory
        cd itgc_app

        # 2. Install dependencies
        pip install -r requirements.txt

        # 3. Place model artefacts
        # Copy your model_artefacts/ folder into itgc_app/

        # 4. Run the app
        streamlit run app.py
        ```

        The app will open at `http://localhost:8501`
        """)

    with tab2:
        st.markdown("""
        ```dockerfile
        # Dockerfile (included in app root)
        FROM python:3.11-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt
        COPY . .
        EXPOSE 8501
        CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
        ```

        ```bash
        # Build and run
        docker build -t itgc-scorer .
        docker run -p 8501:8501 itgc-scorer
        ```
        """)

    with tab3:
        st.markdown("""
        **Azure Container Apps deployment:**

        ```bash
        # 1. Build and push to Azure Container Registry
        az acr build --registry <your-acr> --image itgc-scorer:v1 .

        # 2. Deploy to Container Apps
        az containerapp create \\
          --name itgc-scorer \\
          --resource-group <rg> \\
          --environment <env> \\
          --image <your-acr>.azurecr.io/itgc-scorer:v1 \\
          --target-port 8501 \\
          --ingress external
        ```

        Refer to Azure Container Apps documentation for full configuration including managed identity and Key Vault for secrets.
        """)

    # ── Requirements ──────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📦 requirements.txt"):
        st.code("""streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.44.0
matplotlib>=3.7.0
openpyxl>=3.1.0
joblib>=1.3.0
""", language="text")
