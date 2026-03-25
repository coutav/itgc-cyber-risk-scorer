"""
pages/batch_scoring.py
Batch upload (CSV or Excel), score all findings, display results, download.
"""

import streamlit as st
import pandas as pd
import io

from utils.model_loader import load_artefacts
from utils.inference import predict_risk, band_color, band_bg
from utils.export import batch_to_excel


REQUIRED_COLS = [
    "Observation", "Risk", "Control Domain",
    "Application", "Industry", "Application Type"
]

OPTIONAL_COLS = ["Reference", "Title"]

DOMAIN_OPTIONS = ["PAM", "NJL", "CM", "BR"]
INDUSTRY_OPTIONS = ["Energy & Utilities", "Financial Services",
                    "Manufacturing", "Pharmaceuticals", "Retail"]


def render_band_badge(band: str) -> str:
    c = band_color(band)
    return f'<span style="color:{c};font-weight:600;font-family:\'DM Mono\',monospace">{band}</span>'


def render():
    st.markdown("## 📋 Batch Scoring")
    st.markdown('<div class="section-header">Upload a CSV or Excel file containing multiple ITGC findings to score in bulk</div>', unsafe_allow_html=True)

    artefacts = load_artefacts()
    if not artefacts:
        st.error("Model artefacts not loaded. Place the `model_artefacts/` folder in the app root.")
        return

    # ── Template download ──────────────────────────────────────────────────────
    with st.expander("📥 Download input template", expanded=False):
        st.markdown("Your file must contain these columns (exact names):")
        st.markdown("""
        | Column | Required | Description |
        |---|---|---|
        | `Reference` | Optional | Audit finding reference (e.g. D.2024.1) |
        | `Title` | Optional | Short title of the finding |
        | `Observation` | ✓ | Full observation text |
        | `Risk` | ✓ | Full risk statement |
        | `Control Domain` | ✓ | One of: PAM, NJL, CM, BR |
        | `Application` | ✓ | System / application name |
        | `Industry` | ✓ | One of: Energy & Utilities, Financial Services, Manufacturing, Pharmaceuticals, Retail |
        | `Application Type` | ✓ | Non-Generic or Generic / Home-grown |
        """)

        # Generate blank template
        template_df = pd.DataFrame(columns=OPTIONAL_COLS + REQUIRED_COLS)
        buf = io.BytesIO()
        template_df.to_csv(buf, index=False)
        st.download_button(
            "⬇  Download blank CSV template",
            data=buf.getvalue(),
            file_name="itgc_batch_template.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # ── File uploader ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload findings file",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel file. First row must be column headers.",
    )

    if uploaded is None:
        st.markdown("""
        <div style="background:#111827;border:1px dashed #1e2d45;border-radius:12px;padding:3rem;text-align:center;margin-top:1rem">
          <div style="font-size:2rem;margin-bottom:0.5rem">📂</div>
          <div style="color:#4a5568;font-size:0.9rem">Upload a CSV or Excel file to begin batch scoring</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Load file ──────────────────────────────────────────────────────────────
    try:
        if uploaded.name.endswith(".csv"):
            df_input = pd.read_csv(uploaded)
        else:
            df_input = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    st.markdown(f'<div class="pwc-card"><span style="color:#8898aa;font-size:0.8rem">Loaded</span> <span style="color:#06b6d4;font-family:\'DM Mono\',monospace">{uploaded.name}</span> <span style="color:#4a5568;font-size:0.8rem">— {len(df_input):,} rows × {df_input.shape[1]} columns</span></div>', unsafe_allow_html=True)

    # ── Column validation ──────────────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLS if c not in df_input.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        with st.expander("Columns found in your file"):
            st.write(list(df_input.columns))
        return

    # Preview
    with st.expander("Preview uploaded data (first 5 rows)"):
        st.dataframe(df_input.head(), use_container_width=True)

    # ── Distribution summary before scoring ───────────────────────────────────
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.metric("Total Findings", len(df_input))
    with col_d2:
        domain_counts = df_input["Control Domain"].value_counts()
        top_domain = domain_counts.index[0] if len(domain_counts) > 0 else "—"
        st.metric("Top Domain", top_domain)
    with col_d3:
        n_industries = df_input["Industry"].nunique()
        st.metric("Industries", n_industries)

    st.markdown("---")

    # ── Score button ───────────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_batch = st.button("▶  Score All Findings", use_container_width=True)
    with col_info:
        st.markdown(f'<div style="padding:0.65rem 0;color:#8898aa;font-size:0.82rem">Will score {len(df_input):,} findings using XGBoost + SHAP pipeline</div>', unsafe_allow_html=True)

    if not run_batch:
        return

    # ── Run batch inference ────────────────────────────────────────────────────
    progress_bar = st.progress(0, text="Scoring findings…")
    results_rows = []

    for i, row in df_input.iterrows():
        try:
            r = predict_risk(
                observation    = str(row.get("Observation", "")),
                risk           = str(row.get("Risk", "")),
                control_domain = str(row.get("Control Domain", "PAM")),
                application    = str(row.get("Application", "Unknown")),
                industry       = str(row.get("Industry", "Manufacturing")),
                app_type       = str(row.get("Application Type", "Non-Generic")),
                artefacts      = artefacts,
                compute_shap   = False,
            )
            results_rows.append({
                "Reference":       row.get("Reference", f"ROW-{i+1}"),
                "Title":           row.get("Title", ""),
                "Control Domain":  row.get("Control Domain", ""),
                "Application":     row.get("Application", ""),
                "Industry":        row.get("Industry", ""),
                "Risk Score":      r["risk_score"],
                "Risk Band":       r["risk_band"],
                "Predicted Class": r["predicted_class"],
                "P(High)":         r["p_high"],
                "P(Medium)":       r["p_medium"],
                "P(Low)":          r["p_low"],
            })
        except Exception as e:
            results_rows.append({
                "Reference":       row.get("Reference", f"ROW-{i+1}"),
                "Title":           row.get("Title", ""),
                "Control Domain":  row.get("Control Domain", ""),
                "Application":     row.get("Application", ""),
                "Industry":        row.get("Industry", ""),
                "Risk Score":      None,
                "Risk Band":       "ERROR",
                "Predicted Class": str(e),
                "P(High)":         None,
                "P(Medium)":       None,
                "P(Low)":          None,
            })

        progress_bar.progress((i + 1) / len(df_input),
                               text=f"Scored {i+1}/{len(df_input)} findings…")

    progress_bar.empty()
    df_results = pd.DataFrame(results_rows)
    st.session_state["batch_results"] = df_results
    st.success(f"✓ Scored {len(df_results)} findings successfully.")

    # ── Summary statistics ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Batch Results Summary")

    valid = df_results[df_results["Risk Band"] != "ERROR"]
    band_counts = valid["Risk Band"].value_counts()

    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Scored", len(df_results))
    with cols[1]:
        n_high = band_counts.get("High", 0)
        st.metric("High Risk", n_high, delta=f"{n_high/len(valid)*100:.1f}%" if len(valid) else "—")
    with cols[2]:
        n_med = band_counts.get("Medium", 0)
        st.metric("Medium Risk", n_med, delta=f"{n_med/len(valid)*100:.1f}%" if len(valid) else "—")
    with cols[3]:
        n_low = band_counts.get("Low", 0)
        st.metric("Low Risk", n_low, delta=f"{n_low/len(valid)*100:.1f}%" if len(valid) else "—")

    # Score distribution chart
    import matplotlib.pyplot as plt

    if len(valid) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        fig.patch.set_facecolor("#111827")

        # Histogram of scores
        ax1 = axes[0]
        ax1.set_facecolor("#111827")
        scores = valid["Risk Score"].dropna()
        ax1.hist(scores[valid["Risk Band"] == "Low"],    bins=20, color="#10b981", alpha=0.7, label="Low")
        ax1.hist(scores[valid["Risk Band"] == "Medium"], bins=20, color="#d97706", alpha=0.7, label="Medium")
        ax1.hist(scores[valid["Risk Band"] == "High"],   bins=20, color="#e53e3e", alpha=0.7, label="High")
        ax1.set_xlabel("Risk Score", fontsize=9, color="#8898aa")
        ax1.set_ylabel("Count", fontsize=9, color="#8898aa")
        ax1.set_title("Score Distribution", fontsize=10, color="#c0cce0", fontweight="500")
        ax1.tick_params(colors="#4a5568")
        for spine in ax1.spines.values():
            spine.set_color("#1e2d45")
        ax1.legend(fontsize=8, framealpha=0, labelcolor="#8898aa")

        # Band by domain
        ax2 = axes[1]
        ax2.set_facecolor("#111827")
        domain_band = valid.groupby(["Control Domain", "Risk Band"]).size().unstack(fill_value=0)
        colors_map = {"High": "#e53e3e", "Medium": "#d97706", "Low": "#10b981"}
        x = range(len(domain_band.index))
        bottom = [0] * len(domain_band.index)
        for band_name in ["Low", "Medium", "High"]:
            if band_name in domain_band.columns:
                vals = domain_band[band_name].values
                ax2.bar(x, vals, bottom=bottom, color=colors_map[band_name],
                        label=band_name, alpha=0.85, width=0.6)
                bottom = [b + v for b, v in zip(bottom, vals)]
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(domain_band.index, fontsize=9, color="#8898aa")
        ax2.set_title("Risk Band by Domain", fontsize=10, color="#c0cce0", fontweight="500")
        ax2.tick_params(colors="#4a5568")
        for spine in ax2.spines.values():
            spine.set_color("#1e2d45")
        ax2.legend(fontsize=8, framealpha=0, labelcolor="#8898aa")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Results table ──────────────────────────────────────────────────────────
    st.markdown("### 📋 Scored Findings")

    # Filter controls
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_band = st.multiselect("Filter by band", ["High", "Medium", "Low"],
                                      default=["High", "Medium", "Low"])
    with col_f2:
        filter_domain = st.multiselect("Filter by domain",
                                        df_results["Control Domain"].unique().tolist(),
                                        default=df_results["Control Domain"].unique().tolist())
    with col_f3:
        sort_col = st.selectbox("Sort by", ["Risk Score", "Risk Band", "Control Domain"], index=0)

    filtered = df_results[
        df_results["Risk Band"].isin(filter_band) &
        df_results["Control Domain"].isin(filter_domain)
    ].sort_values(sort_col, ascending=(sort_col != "Risk Score"))

    st.dataframe(
        filtered.style
            .format({"Risk Score": "{:.1f}", "P(High)": "{:.3f}", "P(Medium)": "{:.3f}", "P(Low)": "{:.3f}"})
            .applymap(lambda v: f"color: #e53e3e; font-weight: 600" if v == "High"
                      else (f"color: #d97706; font-weight: 600" if v == "Medium"
                      else (f"color: #10b981; font-weight: 600" if v == "Low" else "")),
                      subset=["Risk Band"]),
        use_container_width=True,
        height=420,
    )

    # ── Download buttons ───────────────────────────────────────────────────────
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_buf = df_results.to_csv(index=False).encode()
        st.download_button("⬇  Download CSV", data=csv_buf,
                           file_name="itgc_batch_scores.csv", mime="text/csv",
                           use_container_width=True)

    with col_dl2:
        try:
            excel_bytes = batch_to_excel(df_results)
            st.download_button("⬇  Download Excel (formatted)", data=excel_bytes,
                               file_name="itgc_batch_scores.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
        except Exception as e:
            st.warning(f"Excel export unavailable: {e}")
