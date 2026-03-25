"""
utils/model_loader.py
Loads and caches all model artefacts needed for inference.
"""

import streamlit as st
import os
import joblib
import re

ARTEFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_artefacts")

REQUIRED_FILES = [
    "xgb_model.pkl",
    "scaler.pkl",
    "label_encoder.pkl",
    "tfidf_vectoriser.pkl",
    "app_tier_map.pkl",
    "keywords.pkl",
]


@st.cache_resource(show_spinner=False)
def load_artefacts():
    """Load all model artefacts into a dict. Returns None if any are missing."""
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(ARTEFACT_DIR, f))]
    if missing:
        return None

    return {
        "model":      joblib.load(os.path.join(ARTEFACT_DIR, "xgb_model.pkl")),
        "scaler":     joblib.load(os.path.join(ARTEFACT_DIR, "scaler.pkl")),
        "le":         joblib.load(os.path.join(ARTEFACT_DIR, "label_encoder.pkl")),
        "tfidf":      joblib.load(os.path.join(ARTEFACT_DIR, "tfidf_vectoriser.pkl")),
        "app_tier":   joblib.load(os.path.join(ARTEFACT_DIR, "app_tier_map.pkl")),
        "keywords":   joblib.load(os.path.join(ARTEFACT_DIR, "keywords.pkl")),
    }
