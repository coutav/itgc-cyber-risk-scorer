"""
pages/model_insights.py
Model performance dashboard — CV results, scoring formula, feature importance overview.
Purely static/informational (no new inference).
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Hardcoded from notebook Phase 4 outputs ───────────────────────────────────
CV_RESULTS = {
    "Logistic Regression": {"CV Macro-F1": 0.637, "CV Macro-F1 std": 0.019,
                             "CV Weighted-F1": 0.642, "CV Micro-F1": 0.643, "Train Macro-F1": 0.648},
    "Decision Tree":       {"CV Macro-F1": 0.641, "CV Macro-F1 std": 0.039,
                             "CV Weighted-F1": 0.646, "CV Micro-F1": 0.648, "Train Macro-F1": 0.765},
    "Random Forest":       {"CV Macro-F1": 0.660, "CV Macro-F1 std": 0.021,
                             "CV Weighted-F1": 0.664, "CV Micro-F1": 0.666, "Train Macro-F1": 0.998},
    "XGBoost ✓":           {"CV Macro-F1": 0.667, "CV Macro-F1 std": 0.032,
                             "CV Weighted-F1": 0.672, "CV Micro-F1": 0.671, "Train Macro-F1": 0.994},
}

# Label distribution from notebook Phase 1
LABEL_DIST = {"High": 415, "Medium": 227, "Low": 358}
PRIORITY_DIST = {1: 163, 2: 195, 3: 227, 4: 271, 5: 144}

# Real validation results from Phase 6 (23 rows)
REAL_VALIDATION = {
    "Exact match": 15,
    "Adjacent (±1 band)": 7,
    "Mismatch": 1,
    "Total": 23,
}


def render_cv_bars():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    models  = list(CV_RESULTS.keys())
    f1s     = [CV_RESULTS[m]["CV Macro-F1"] for m in models]
    stds    = [CV_RESULTS[m]["CV Macro-F1 std"] for m in models]
    colors  = ["#4a5568", "#4a5568", "#4a5568", "#06b6d4"]

    x = np.arange(len(models))
    bars = ax.bar(x, f1s, color=colors, width=0.55, yerr=stds,
                  capsize=5, error_kw={"elinewidth": 1.5, "ecolor": "#8898aa"},
                  alpha=0.9, edgecolor="none")

    for bar, v, std in zip(bars, f1s, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, v + std + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                color="#c0cce0", fontfamily="DejaVu Sans")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9.5, color="#8898aa")
    ax.set_ylabel("CV Macro-F1", fontsize=9, color="#8898aa")
    ax.set_ylim(0.58, 0.74)
    ax.set_title("5-Fold Cross-Validation Macro-F1 Comparison", fontsize=11,
                 color="#c0cce0", fontweight="500", pad=10)
    ax.tick_params(colors="#4a5568")
    for spine in ax.spines.values():
        spine.set_color("#1e2d45")

    # Highlight best
    ax.axhline(max(f1s), color="#06b6d4", linestyle="--", linewidth=0.8, alpha=0.4)

    plt.tight_layout()
    return fig


def render_label_donut():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.patch.set_facecolor("#111827")

    # 3-class donut
    ax1 = axes[0]
    ax1.set_facecolor("#111827")
    labels = ["High", "Medium", "Low"]
    sizes  = [LABEL_DIST[l] for l in labels]
    colors = ["#e53e3e", "#d97706", "#10b981"]
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        pctdistance=0.75, startangle=90,
        wedgeprops=dict(width=0.48, edgecolor="#0a0e1a", linewidth=2)
    )
    for t in texts:
        t.set_fontsize(9); t.set_color("#8898aa")
    for at in autotexts:
        at.set_fontsize(8); at.set_color("#f0f4ff")
    ax1.set_title("3-Class Label Distribution\n(n=1,000)", fontsize=9.5,
                  color="#c0cce0", pad=6)

    # Priority bar
    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    ps     = list(PRIORITY_DIST.keys())
    ns     = list(PRIORITY_DIST.values())
    pcols  = ["#e53e3e", "#e97a0d", "#d97706", "#6b8e23", "#10b981"]
    bars   = ax2.bar([f"P{p}" for p in ps], ns, color=pcols, width=0.55, alpha=0.9)
    for bar, n in zip(bars, ns):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                 str(n), ha="center", fontsize=8, color="#8898aa")
    ax2.set_title("Raw Priority Distribution (P1–P5)", fontsize=9.5,
                  color="#c0cce0", pad=6)
    ax2.set_ylabel("Count", fontsize=8, color="#8898aa")
    ax2.tick_params(colors="#4a5568")
    for spine in ax2.spines.values():
        spine.set_color("#1e2d45")

    plt.tight_layout()
    return fig


def render_scoring_formula():
    st.markdown("""
    <div class="pwc-card">
    <div class="section-header">Risk Score Formula</div>
    <div style="font-family:'DM Mono',monospace;font-size:1.05rem;color:#06b6d4;padding:0.8rem 0">
      Score = P(Low) × 20 + P(Medium) × 55 + P(High) × 90
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:0.5rem">
      <div style="background:#1a2236;border-radius:8px;padding:0.8rem;border:1px solid #1e2d45">
        <div style="color:#10b981;font-size:1.4rem;font-weight:600;font-family:'DM Mono',monospace">×20</div>
        <div style="color:#8898aa;font-size:0.78rem;margin-top:0.2rem">Low severity anchor<br>Score range: 0–20</div>
      </div>
      <div style="background:#1a2236;border-radius:8px;padding:0.8rem;border:1px solid #1e2d45">
        <div style="color:#d97706;font-size:1.4rem;font-weight:600;font-family:'DM Mono',monospace">×55</div>
        <div style="color:#8898aa;font-size:0.78rem;margin-top:0.2rem">Medium severity anchor<br>Score range: 20–55</div>
      </div>
      <div style="background:#1a2236;border-radius:8px;padding:0.8rem;border:1px solid #1e2d45">
        <div style="color:#e53e3e;font-size:1.4rem;font-weight:600;font-family:'DM Mono',monospace">×90</div>
        <div style="color:#8898aa;font-size:0.78rem;margin-top:0.2rem">High severity anchor<br>Score range: 55–100</div>
      </div>
    </div>
    <div style="margin-top:1rem;display:grid;grid-template-columns:1fr 1fr;gap:0.8rem">
      <div>
        <div class="section-header" style="margin-bottom:0.3rem">Band Thresholds</div>
        <div class="info-row"><span class="info-label">Low</span><span class="info-value" style="color:#10b981">0 – 39.9</span></div>
        <div class="info-row"><span class="info-label">Medium</span><span class="info-value" style="color:#d97706">40 – 69.9</span></div>
        <div class="info-row"><span class="info-label">High</span><span class="info-value" style="color:#e53e3e">70 – 100</span></div>
      </div>
      <div>
        <div class="section-header" style="margin-bottom:0.3rem">Label Mapping</div>
        <div class="info-row"><span class="info-label">Priority 1–2</span><span class="info-value" style="color:#e53e3e">High</span></div>
        <div class="info-row"><span class="info-label">Priority 3</span><span class="info-value" style="color:#d97706">Medium</span></div>
        <div class="info-row"><span class="info-label">Priority 4–5</span><span class="info-value" style="color:#10b981">Low</span></div>
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)


def render():
    st.markdown("## 📊 Model Insights")
    st.markdown('<div class="section-header">Performance metrics, scoring formula, and dataset statistics from the thesis model pipeline</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 CV Performance", "🗂 Dataset Stats", "⚙️ Scoring Formula", "✅ Real Validation"])

    with tab1:
        st.markdown("### Cross-Validation Results")
        st.markdown('<div class="section-header">Stratified 5-fold CV on 977 synthetic training rows — primary metric: Macro-F1</div>', unsafe_allow_html=True)

        fig = render_cv_bars()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("---")

        import pandas as pd
        cv_df = pd.DataFrame(CV_RESULTS).T.reset_index()
        cv_df.columns = ["Model", "Train Macro-F1", "CV Macro-F1", "CV Macro-F1 Std", "CV Weighted-F1", "CV Micro-F1"]
        cv_df = cv_df[["Model", "CV Macro-F1", "CV Macro-F1 Std", "CV Weighted-F1", "CV Micro-F1", "Train Macro-F1"]]

        st.dataframe(
            cv_df.style
                .format({c: "{:.3f}" for c in cv_df.columns if c != "Model"})
                .applymap(lambda v: "color: #06b6d4; font-weight: 600" if isinstance(v, float) and v == cv_df["CV Macro-F1"].max() else "", subset=["CV Macro-F1"]),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("""
        <div style="margin-top:1rem;padding:0.8rem 1rem;background:#111827;border:1px solid #1e2d45;border-radius:10px;font-size:0.82rem;color:#8898aa">
        <b style="color:#06b6d4">Why Macro-F1?</b> Macro-F1 weights all three severity classes equally, penalising models that ignore the minority Medium class.
        In an audit context, missing a High-severity finding is the worst-case error — Macro-F1 correctly captures this without allowing the model to hide behind class imbalance.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Dataset Statistics")
        st.markdown('<div class="section-header">Combined_Deficiencies_v8 · 1,000 rows · 10 columns · 0 nulls</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Rows", "1,000")
        with col2: st.metric("Synthetic", "977")
        with col3: st.metric("Real (Held-out)", "23")
        with col4: st.metric("Control Domains", "4")

        fig2 = render_label_donut()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown("---")
        st.markdown('<div class="section-header">Feature Matrix Composition</div>', unsafe_allow_html=True)
        import pandas as pd
        feat_df = pd.DataFrame([
            {"Feature Group": "Text length features", "Count": 4, "Examples": "obs_char_len, risk_char_len, obs_word_count, obs_bullet_count"},
            {"Feature Group": "High-severity keyword count", "Count": 1, "Examples": "high_sev_kw_count"},
            {"Feature Group": "Quantity finding flag", "Count": 1, "Examples": "has_quantity_finding"},
            {"Feature Group": "Cyber risk binary flags", "Count": 5, "Examples": "flag_unauth_access, flag_data_loss, …"},
            {"Feature Group": "Control domain one-hot", "Count": 4, "Examples": "domain_PAM, domain_NJL, domain_CM, domain_BR"},
            {"Feature Group": "Industry one-hot", "Count": 5, "Examples": "industry_Financial Services, …"},
            {"Feature Group": "Application type one-hot", "Count": 2, "Examples": "apptype_Non-Generic, …"},
            {"Feature Group": "Application tier", "Count": 1, "Examples": "app_tier (1=Tier1, 3=unknown)"},
            {"Feature Group": "TF-IDF tokens", "Count": 100, "Examples": "tfidf_access, tfidf_user, …"},
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
        st.markdown('<div style="font-size:0.78rem;color:#4a5568;margin-top:0.3rem">Total feature dimensions: 123 (after StandardScaler normalisation)</div>', unsafe_allow_html=True)

    with tab3:
        render_scoring_formula()

        st.markdown("---")
        st.markdown("### XGBoost Configuration")
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.82rem;background:#111827;border:1px solid #1e2d45;border-radius:10px;padding:1rem;color:#c0cce0">
n_estimators    = 200<br>
eval_metric     = mlogloss<br>
random_state    = 42<br>
n_jobs          = -1<br>
verbosity       = 0<br>
class_weight    = balanced (via sample_weight)<br>
cv              = StratifiedKFold(n_splits=5)
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### Real-Data Validation (Phase 6)")
        st.markdown('<div class="section-header">23 real ITGC deficiency findings held out from training — never seen during model development</div>', unsafe_allow_html=True)

        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            pct = REAL_VALIDATION["Exact match"] / REAL_VALIDATION["Total"] * 100
            st.metric("Exact Match", f"{REAL_VALIDATION['Exact match']}/23", delta=f"{pct:.1f}%")
        with col_v2:
            pct2 = REAL_VALIDATION["Adjacent (±1 band)"] / REAL_VALIDATION["Total"] * 100
            st.metric("Adjacent (±1 band)", f"{REAL_VALIDATION['Adjacent (±1 band)']}/23", delta=f"{pct2:.1f}%")
        with col_v3:
            pct3 = REAL_VALIDATION["Mismatch"] / REAL_VALIDATION["Total"] * 100
            st.metric("Mismatch", f"{REAL_VALIDATION['Mismatch']}/23", delta=f"{pct3:.1f}%")

        # Visual bar
        fig_v, ax_v = plt.subplots(figsize=(7, 1.8))
        fig_v.patch.set_facecolor("#111827")
        ax_v.set_facecolor("#111827")

        categories = ["Exact Match", "Adjacent ±1", "Mismatch"]
        values     = [15, 7, 1]
        colors_v   = ["#10b981", "#d97706", "#e53e3e"]
        left = 0
        for cat, val, col in zip(categories, values, colors_v):
            ax_v.barh(0, val, left=left, color=col, height=0.55, alpha=0.85)
            if val > 0:
                ax_v.text(left + val / 2, 0, f"{cat}\n{val}", ha="center", va="center",
                          fontsize=8.5, color="white", fontfamily="DejaVu Sans", fontweight="600")
            left += val

        ax_v.set_xlim(0, 23)
        ax_v.set_yticks([])
        ax_v.set_xticks(range(0, 24, 5))
        ax_v.tick_params(colors="#4a5568", labelsize=8)
        for spine in ax_v.spines.values():
            spine.set_color("#1e2d45")
        ax_v.set_title("Real Validation Results (n=23)", fontsize=9.5, color="#c0cce0", pad=8)
        plt.tight_layout()
        st.pyplot(fig_v, use_container_width=True)
        plt.close()

        st.markdown("""
        <div style="margin-top:1rem;padding:0.8rem 1rem;background:#111827;border:1px solid #1e2d45;border-radius:10px;font-size:0.82rem;color:#8898aa">
        <b style="color:#10b981">Validation sources:</b> Def_1.xlsx and Def_2.xlsx — 23 real anonymised ITGC deficiency findings from PwC Denmark engagements.
        These findings were separated before any model training and used exclusively for Phase 6 evaluation.
        </div>
        """, unsafe_allow_html=True)
