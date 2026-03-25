"""
utils/inference.py
Mirrors the predict_risk() function from the thesis notebook exactly.
Produces risk_score (0–100), risk_band, predicted_class, probabilities, and SHAP values.
"""

import pandas as pd
import numpy as np
import re


# ── Regex pattern for quantity findings (matches notebook) ────────────────────
QUANTITY_PATTERN = re.compile(
    r'\b\d+\s*(user|account|access|log|record|instance|case|employee|request)',
    re.IGNORECASE
)

# ── Band assignment matching notebook thresholds ──────────────────────────────
def assign_band(score: float) -> str:
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


def band_color(band: str) -> str:
    return {"High": "#e53e3e", "Medium": "#d97706", "Low": "#10b981"}.get(band, "#8898aa")


def band_bg(band: str) -> str:
    return {
        "High":   "rgba(229,62,62,0.12)",
        "Medium": "rgba(217,119,6,0.12)",
        "Low":    "rgba(16,185,129,0.12)",
    }.get(band, "rgba(74,85,104,0.12)")


# ── All valid categorical values (must match training one-hot schema) ─────────
ALL_DOMAINS    = ['BR', 'CM', 'NJL', 'PAM']
ALL_INDUSTRIES = ['Energy & Utilities', 'Financial Services',
                  'Manufacturing', 'Pharmaceuticals', 'Retail']
ALL_APPTYPES   = ['Generic / Home-grown', 'Non-Generic']


def build_features(observation: str, risk: str, control_domain: str,
                   application: str, industry: str, app_type: str,
                   app_tier_map: dict, keywords: list) -> pd.DataFrame:
    """Build the structured + one-hot feature row matching training schema."""

    row = pd.DataFrame([{
        'Observation':     observation,
        'Risk':            risk,
        'Control Domain':  control_domain,
        'Application':     application,
        'Industry':        industry,
        'Application Type': app_type,
    }])

    combined = (row['Observation'] + ' ' + row['Risk']).str.lower()

    feats = pd.DataFrame()
    feats['obs_char_len']         = row['Observation'].str.len()
    feats['risk_char_len']        = row['Risk'].str.len()
    feats['obs_word_count']       = row['Observation'].str.split().str.len()
    feats['obs_bullet_count']     = row['Observation'].str.count(r'\n')
    feats['high_sev_kw_count']    = combined.apply(
        lambda t: sum(1 for kw in keywords if kw in t))
    feats['has_quantity_finding'] = combined.apply(
        lambda t: 1 if QUANTITY_PATTERN.search(t) else 0)
    feats['flag_unauth_access']   = combined.str.contains(
        r'unauthori[sz]ed access|former employ|post.termination|active login',
        regex=True).astype(int)
    feats['flag_data_loss']       = combined.str.contains(
        r'data loss|alterat|erroneous|integrity|financial posting',
        regex=True).astype(int)
    feats['flag_priv_escalation'] = combined.str.contains(
        r'privilege|escalat|admin|segregation of duties|sod violation',
        regex=True).astype(int)
    feats['flag_no_logging']      = combined.str.contains(
        r'audit trail|logging|traceability|incident detect|no auditing',
        regex=True).astype(int)
    feats['flag_weak_credentials'] = combined.str.contains(
        r'brute.?force|password|credential|default pass|weak pass',
        regex=True).astype(int)

    for d in ALL_DOMAINS:
        feats[f'domain_{d}'] = int(control_domain == d)
    for ind in ALL_INDUSTRIES:
        feats[f'industry_{ind}'] = int(industry == ind)
    for at in ALL_APPTYPES:
        feats[f'apptype_{at}'] = int(app_type == at)

    feats['app_tier'] = app_tier_map.get(application, 2)
    feats = feats.reset_index(drop=True)
    return feats, row


def predict_risk(observation: str, risk: str, control_domain: str,
                 application: str, industry: str, app_type: str,
                 artefacts: dict, compute_shap: bool = True) -> dict:
    """
    Score a single ITGC deficiency finding.

    Returns dict with:
        risk_score, risk_band, predicted_class,
        p_low, p_medium, p_high,
        flags (dict of flag_* columns),
        shap_values (dict feature->shap) if compute_shap=True
    """
    import shap as shap_lib

    model     = artefacts["model"]
    scaler    = artefacts["scaler"]
    le        = artefacts["le"]
    tfidf     = artefacts["tfidf"]
    app_tier  = artefacts["app_tier"]
    keywords  = artefacts["keywords"]

    feats, row = build_features(
        observation, risk, control_domain,
        application, industry, app_type,
        app_tier, keywords
    )

    # TF-IDF
    text_combined = row['Observation'] + ' ' + row['Risk']
    tfidf_vec     = tfidf.transform(text_combined)
    tfidf_df      = pd.DataFrame(
        tfidf_vec.toarray(),
        columns=[f'tfidf_{t}' for t in tfidf.get_feature_names_out()]
    )

    X_new = pd.concat([feats, tfidf_df], axis=1)
    X_new.columns = X_new.columns.astype(str)

    # Align to training columns using scaler's feature_names_in_
    train_cols = scaler.feature_names_in_
    X_new = X_new.reindex(columns=train_cols, fill_value=0)

    X_scaled = pd.DataFrame(
        scaler.transform(X_new),
        columns=X_new.columns
    )

    # Predict
    probs      = model.predict_proba(X_scaled)[0]
    pred_enc   = model.predict(X_scaled)[0]
    pred_class = le.inverse_transform([pred_enc])[0]

    classes = list(le.classes_)
    idx_low    = classes.index('Low')
    idx_medium = classes.index('Medium')
    idx_high   = classes.index('High')

    p_low    = float(probs[idx_low])
    p_medium = float(probs[idx_medium])
    p_high   = float(probs[idx_high])

    score = p_low * 20 + p_medium * 55 + p_high * 90
    band  = assign_band(score)

    # Flags dict for UI display
    flag_cols = [
        'flag_unauth_access', 'flag_data_loss', 'flag_priv_escalation',
        'flag_no_logging', 'flag_weak_credentials'
    ]
    flags = {col: int(feats[col].iloc[0]) for col in flag_cols}

    result = {
        'risk_score':      round(score, 1),
        'risk_band':       band,
        'predicted_class': pred_class,
        'p_low':           round(p_low,    3),
        'p_medium':        round(p_medium, 3),
        'p_high':          round(p_high,   3),
        'flags':           flags,
        'obs_word_count':  int(feats['obs_word_count'].iloc[0]),
        'high_sev_kw_count': int(feats['high_sev_kw_count'].iloc[0]),
        'has_quantity_finding': int(feats['has_quantity_finding'].iloc[0]),
        'app_tier':        int(feats['app_tier'].iloc[0]),
    }

    # SHAP explanation
    if compute_shap:
        try:
            explainer    = shap_lib.TreeExplainer(model)
            shap_vals    = explainer.shap_values(X_scaled)
            # shap_vals shape: (n_classes, n_samples, n_features) for XGBoost multi-class
            # We want the High class contributions for the single row
            if isinstance(shap_vals, list):
                sv_high = shap_vals[idx_high][0]
            else:
                sv_high = shap_vals[0, :, idx_high] if shap_vals.ndim == 3 else shap_vals[0]

            feature_names = list(X_scaled.columns)
            # Keep only non-tfidf features for display (top structural features)
            structural_mask = [not n.startswith('tfidf_') for n in feature_names]
            struct_names  = [n for n, m in zip(feature_names, structural_mask) if m]
            struct_vals   = [v for v, m in zip(sv_high, structural_mask) if m]

            # Also grab top 5 tfidf by abs magnitude
            tfidf_mask  = [n.startswith('tfidf_') for n in feature_names]
            tfidf_names = [n for n, m in zip(feature_names, tfidf_mask) if m]
            tfidf_vals  = [v for v, m in zip(sv_high, tfidf_mask) if m]
            if tfidf_vals:
                top_tfidf_idx = sorted(range(len(tfidf_vals)), key=lambda i: abs(tfidf_vals[i]), reverse=True)[:5]
                top_tfidf_n   = [tfidf_names[i] for i in top_tfidf_idx]
                top_tfidf_v   = [tfidf_vals[i]  for i in top_tfidf_idx]
            else:
                top_tfidf_n, top_tfidf_v = [], []

            all_names = struct_names + top_tfidf_n
            all_vals  = struct_vals  + top_tfidf_v

            # Sort by absolute SHAP value
            sorted_pairs = sorted(zip(all_names, all_vals), key=lambda x: abs(x[1]), reverse=True)
            result['shap_features'] = [p[0] for p in sorted_pairs]
            result['shap_values']   = [p[1] for p in sorted_pairs]
        except Exception as e:
            result['shap_features'] = []
            result['shap_values']   = []
            result['shap_error']    = str(e)

    return result


def batch_predict(df_input: pd.DataFrame, artefacts: dict) -> pd.DataFrame:
    """Score a DataFrame of findings. Expected columns match the raw dataset schema."""
    required = ['Observation', 'Risk', 'Control Domain', 'Application', 'Industry', 'Application Type']
    missing  = [c for c in required if c not in df_input.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    results = []
    for _, row in df_input.iterrows():
        try:
            r = predict_risk(
                observation    = str(row.get('Observation', '')),
                risk           = str(row.get('Risk', '')),
                control_domain = str(row.get('Control Domain', 'PAM')),
                application    = str(row.get('Application', 'Unknown')),
                industry       = str(row.get('Industry', 'Manufacturing')),
                app_type       = str(row.get('Application Type', 'Non-Generic')),
                artefacts      = artefacts,
                compute_shap   = False,
            )
            results.append({
                'Reference':       row.get('Reference', ''),
                'Title':           row.get('Title', ''),
                'Control Domain':  row.get('Control Domain', ''),
                'Application':     row.get('Application', ''),
                'Industry':        row.get('Industry', ''),
                'Risk Score':      r['risk_score'],
                'Risk Band':       r['risk_band'],
                'Predicted Class': r['predicted_class'],
                'P(Low)':          r['p_low'],
                'P(Medium)':       r['p_medium'],
                'P(High)':         r['p_high'],
            })
        except Exception as e:
            results.append({
                'Reference':       row.get('Reference', ''),
                'Title':           row.get('Title', ''),
                'Control Domain':  row.get('Control Domain', ''),
                'Application':     row.get('Application', ''),
                'Industry':        row.get('Industry', ''),
                'Risk Score':      None,
                'Risk Band':       f'ERROR: {e}',
                'Predicted Class': None,
                'P(Low)':          None,
                'P(Medium)':       None,
                'P(High)':         None,
            })

    return pd.DataFrame(results)
