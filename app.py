from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "app_config.json"
BUNDLE_PATH = APP_DIR / "modeles" / "modele_acces_soins_bundle.joblib"

THEME_PALETTES = {
    "light": {
        "bg": "#f0f4f8",
        "bg_elevated": "#ffffff",
        "bg_muted": "#dce6f0",
        "bg_sidebar": "#0a2540",
        "text": "#0f1c2e",
        "text_muted": "#3d5268",
        "text_sidebar": "#f1f5f9",
        "border": "rgba(15, 28, 46, 0.14)",
        "accent": "#0e7490",
        "accent_soft": "#d5f5f6",
        "accent_deep": "#0b4f6c",
        "favorable": "#0f766e",
        "favorable_bg": "#dcf7ef",
        "vulnerable": "#be123c",
        "vulnerable_bg": "#ffe4e8",
        "gauge_low": "#fecdd3",
        "gauge_high": "#99f6e4",
        "chart_pos": "#0f766e",
        "chart_neg": "#be123c",
        "plot_font": "#0f1c2e",
        "plot_grid": "rgba(15, 28, 46, 0.10)",
        "hero_from": "#0b4f6c",
        "hero_to": "#0e7490",
        "chip_bg": "#cce7f6",
        "chip_text": "#0a3d5c",
        "metric_bg": "#ffffff",
        "shadow": "0 10px 30px rgba(15, 40, 70, 0.08)",
        "btn_bg": "#ffffff",
        "btn_text": "#0a2540",
        "btn_border": "#0e7490",
        "btn_primary_bg": "#0e7490",
        "btn_primary_text": "#ffffff",
        "btn_sidebar_bg": "#1a4a6e",
        "btn_sidebar_text": "#ffffff",
        "btn_sidebar_border": "#7dd3fc",
    },
    "dark": {
        "bg": "#0a0f1a",
        "bg_elevated": "#152033",
        "bg_muted": "#1e2d45",
        "bg_sidebar": "#050a12",
        "text": "#f1f5f9",
        "text_muted": "#b6c4d8",
        "text_sidebar": "#f1f5f9",
        "border": "rgba(226, 232, 240, 0.18)",
        "accent": "#38bdf8",
        "accent_soft": "rgba(56, 189, 248, 0.16)",
        "accent_deep": "#7dd3fc",
        "favorable": "#2dd4bf",
        "favorable_bg": "rgba(45, 212, 191, 0.16)",
        "vulnerable": "#fb7185",
        "vulnerable_bg": "rgba(251, 113, 133, 0.16)",
        "gauge_low": "rgba(251, 113, 133, 0.25)",
        "gauge_high": "rgba(45, 212, 191, 0.25)",
        "chart_pos": "#2dd4bf",
        "chart_neg": "#fb7185",
        "plot_font": "#f1f5f9",
        "plot_grid": "rgba(226, 232, 240, 0.14)",
        "hero_from": "#0c4a6e",
        "hero_to": "#155e75",
        "chip_bg": "rgba(56, 189, 248, 0.18)",
        "chip_text": "#e0f2fe",
        "metric_bg": "#152033",
        "shadow": "0 12px 36px rgba(0, 0, 0, 0.45)",
        "btn_bg": "#243b55",
        "btn_text": "#ffffff",
        "btn_border": "#38bdf8",
        "btn_primary_bg": "#0284c7",
        "btn_primary_text": "#ffffff",
        "btn_sidebar_bg": "#1e3a5f",
        "btn_sidebar_text": "#ffffff",
        "btn_sidebar_border": "#7dd3fc",
    },
}


st.set_page_config(
    page_title="Prédiction du recours formel aux soins",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def build_theme_css(theme: str) -> str:
    p = THEME_PALETTES[theme]
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

:root {{
    --bg: {p["bg"]};
    --bg-elevated: {p["bg_elevated"]};
    --bg-muted: {p["bg_muted"]};
    --bg-sidebar: {p["bg_sidebar"]};
    --text: {p["text"]};
    --text-muted: {p["text_muted"]};
    --text-sidebar: {p["text_sidebar"]};
    --border: {p["border"]};
    --accent: {p["accent"]};
    --accent-soft: {p["accent_soft"]};
    --accent-deep: {p["accent_deep"]};
    --favorable: {p["favorable"]};
    --favorable-bg: {p["favorable_bg"]};
    --vulnerable: {p["vulnerable"]};
    --vulnerable-bg: {p["vulnerable_bg"]};
    --chip-bg: {p["chip_bg"]};
    --chip-text: {p["chip_text"]};
    --metric-bg: {p["metric_bg"]};
    --shadow: {p["shadow"]};
    --hero-from: {p["hero_from"]};
    --hero-to: {p["hero_to"]};
    --btn-bg: {p["btn_bg"]};
    --btn-text: {p["btn_text"]};
    --btn-border: {p["btn_border"]};
    --btn-primary-bg: {p["btn_primary_bg"]};
    --btn-primary-text: {p["btn_primary_text"]};
    --btn-sidebar-bg: {p["btn_sidebar_bg"]};
    --btn-sidebar-text: {p["btn_sidebar_text"]};
    --btn-sidebar-border: {p["btn_sidebar_border"]};
}}

html, body, [class*="css"] {{
    font-family: "DM Sans", sans-serif;
}}

.stApp {{
    background: var(--bg) !important;
    color: var(--text) !important;
}}

.stApp [data-testid="stAppViewContainer"] {{
    background: var(--bg) !important;
    color: var(--text) !important;
}}

.main .block-container {{
    color: var(--text);
}}

/* Titres et paragraphes toujours contrastés */
h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown strong, .stMarkdown em {{
    color: var(--text) !important;
}}

.hero, .hero h1, .hero p {{
    color: #f8fafc !important;
}}

.brand-mark .brand-text h2 {{
    color: #f8fafc !important;
}}

.brand-mark .brand-text p {{
    color: #94a3b8 !important;
}}

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
label[data-testid="stWidgetLabel"] {{
    color: var(--text) !important;
    opacity: 1 !important;
}}

[data-testid="stSidebar"] {{
    background: var(--bg-sidebar) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}}

[data-testid="stSidebar"] > div:first-child {{
    padding-top: 1rem;
}}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4,
[data-testid="stSidebar"] .stMarkdown h5,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
    color: var(--text-sidebar) !important;
}}

[data-testid="stSidebar"] [data-testid="stMetricValue"],
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {{
    color: #f0f7fc !important;
}}

[data-testid="stSidebar"] div[data-testid="stMetric"] {{
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 0.85rem 1rem;
}}

/* Boutons : contraste permanent (sans survol) */
.stButton > button,
.stDownloadButton > button,
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"],
button[kind="secondary"],
button[kind="primary"] {{
    background-color: var(--btn-bg) !important;
    color: var(--btn-text) !important;
    border: 1.5px solid var(--btn-border) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}}

.stButton > button p,
.stButton > button span,
.stDownloadButton > button p,
.stDownloadButton > button span,
button[data-testid="baseButton-secondary"] p,
button[data-testid="baseButton-secondary"] span,
button[data-testid="baseButton-primary"] p,
button[data-testid="baseButton-primary"] span,
button[kind="secondary"] p,
button[kind="secondary"] span,
button[kind="primary"] p,
button[kind="primary"] span {{
    color: inherit !important;
    opacity: 1 !important;
    visibility: visible !important;
}}

.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"],
button[data-testid="baseButton-primary"],
button[kind="primary"] {{
    background: var(--btn-primary-bg) !important;
    background-image: none !important;
    color: var(--btn-primary-text) !important;
    border: 1.5px solid var(--btn-primary-bg) !important;
    box-shadow: 0 6px 16px rgba(14, 116, 144, 0.25);
}}

.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stDownloadButton > button[kind="primary"] p,
.stDownloadButton > button[kind="primary"] span,
button[data-testid="baseButton-primary"] p,
button[data-testid="baseButton-primary"] span {{
    color: var(--btn-primary-text) !important;
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
    filter: brightness(1.06);
    opacity: 1 !important;
}}

.stButton > button:hover p,
.stButton > button:hover span,
.stDownloadButton > button:hover p,
.stDownloadButton > button:hover span {{
    color: inherit !important;
    opacity: 1 !important;
}}

/* Boutons de la sidebar : fond contrasté + texte blanc toujours lisible */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button {{
    background: var(--btn-sidebar-bg) !important;
    color: var(--btn-sidebar-text) !important;
    border: 1.5px solid var(--btn-sidebar-border) !important;
}}

[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stDownloadButton > button p,
[data-testid="stSidebar"] .stDownloadButton > button span {{
    color: var(--btn-sidebar-text) !important;
    opacity: 1 !important;
}}

.brand-mark {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    animation: fadeSlide 0.45s ease-out;
}}

.brand-mark .logo {{
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, #22d3ee, #0e7490);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    box-shadow: 0 8px 20px rgba(34, 211, 238, 0.25);
}}

.brand-mark .brand-text h2 {{
    margin: 0;
    font-family: "Fraunces", Georgia, serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #f8fafc !important;
    letter-spacing: -0.02em;
}}

.brand-mark .brand-text p {{
    margin: 0.15rem 0 0 0;
    font-size: 0.78rem;
    color: #94a3b8 !important;
}}

.hero {{
    position: relative;
    overflow: hidden;
    padding: 1.9rem 2.1rem;
    border-radius: 20px;
    color: #f8fafc;
    background: linear-gradient(125deg, var(--hero-from) 0%, var(--hero-to) 100%);
    box-shadow: var(--shadow);
    margin-bottom: 1.1rem;
    animation: fadeSlide 0.5s ease-out;
}}

.hero::before {{
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 90% 20%, rgba(255,255,255,0.14), transparent 35%),
        radial-gradient(circle at 10% 90%, rgba(34, 211, 238, 0.18), transparent 40%);
    pointer-events: none;
}}

.hero > * {{
    position: relative;
    z-index: 1;
}}

.hero h1 {{
    font-family: "Fraunces", Georgia, serif;
    font-size: clamp(1.55rem, 2.4vw, 2.15rem);
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
    letter-spacing: -0.02em;
    font-weight: 700;
}}

.hero p {{
    margin: 0;
    font-size: 1.02rem;
    opacity: 0.92;
    max-width: 52rem;
    line-height: 1.55;
}}

.steps {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin: 1rem 0 1.35rem 0;
    animation: fadeSlide 0.55s ease-out 0.05s both;
}}

.step {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.45rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--bg-elevated);
    color: var(--text-muted);
    transition: transform 0.2s ease, border-color 0.2s ease, color 0.2s ease;
}}

.step.active {{
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--accent-deep);
}}

.step .num {{
    width: 1.35rem;
    height: 1.35rem;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    background: var(--bg-muted);
    color: var(--text);
}}

.step.active .num {{
    background: var(--accent);
    color: #041018;
}}

.status-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 1.35rem;
    animation: fadeSlide 0.55s ease-out 0.1s both;
}}

.status-chip {{
    display: inline-flex;
    align-items: center;
    padding: 0.38rem 0.75rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.8rem;
    background: var(--chip-bg);
    color: var(--chip-text);
    border: 1px solid transparent;
    letter-spacing: 0.01em;
}}

.section-head {{
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin: 0.4rem 0 0.85rem 0;
}}

.section-head h2 {{
    font-family: "Fraunces", Georgia, serif;
    font-size: 1.35rem;
    margin: 0;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}}

.section-head .badge {{
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--accent-deep) !important;
    background: var(--accent-soft);
    padding: 0.28rem 0.55rem;
    border-radius: 6px;
}}

.result-banner {{
    border-radius: 18px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
    animation: resultPop 0.55s cubic-bezier(0.22, 1, 0.36, 1);
}}

.result-favorable {{
    background: var(--favorable-bg);
    border-left: 5px solid var(--favorable);
}}

.result-vulnerable {{
    background: var(--vulnerable-bg);
    border-left: 5px solid var(--vulnerable);
}}

.result-banner h3 {{
    margin: 0 0 0.35rem 0;
    font-family: "Fraunces", Georgia, serif;
    font-size: 1.45rem;
    color: var(--text) !important;
}}

.result-banner .level {{
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.35rem;
}}

.result-favorable .level {{ color: var(--favorable) !important; }}
.result-vulnerable .level {{ color: var(--vulnerable) !important; }}

.result-banner p {{
    margin: 0;
    color: var(--text) !important;
    line-height: 1.5;
}}

.result-banner strong {{
    color: var(--text) !important;
}}

.progress-track {{
    height: 10px;
    border-radius: 999px;
    background: var(--bg-muted);
    overflow: hidden;
    margin-top: 0.85rem;
}}

.progress-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}}

.progress-fill.good {{
    background: linear-gradient(90deg, #0f766e, #14b8a6);
}}

.progress-fill.risk {{
    background: linear-gradient(90deg, #be123c, #fb7185);
}}

.small-note {{
    font-size: 0.88rem;
    color: var(--text-muted);
    line-height: 1.5;
}}

div[data-testid="stMetric"] {{
    background: var(--metric-bg);
    border: 1px solid var(--border);
    padding: 0.95rem 1rem;
    border-radius: 16px;
    box-shadow: var(--shadow);
}}

div[data-testid="stMetric"] label {{
    color: var(--text-muted) !important;
}}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: var(--text) !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0.35rem;
    background: var(--bg-muted);
    padding: 0.35rem;
    border-radius: 14px;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
    font-weight: 600;
    color: var(--text-muted);
}}

.stTabs [aria-selected="true"] {{
    background: var(--bg-elevated) !important;
    color: var(--accent-deep) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}

.stButton > button[kind="primary"] {{
    letter-spacing: 0.01em;
    border-radius: 12px !important;
    padding: 0.7rem 1.2rem !important;
    transition: transform 0.15s ease, filter 0.15s ease !important;
}}

.stButton > button[kind="primary"]:hover {{
    transform: translateY(-1px);
    filter: brightness(1.08);
}}

/* Onglets : texte toujours lisible */
.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] p {{
    color: inherit !important;
    opacity: 1 !important;
}}

.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] p {{
    color: var(--accent-deep) !important;
}}

div[data-testid="stExpander"] {{
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 14px;
}}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary span,
div[data-testid="stExpander"] summary p {{
    color: var(--text) !important;
    opacity: 1 !important;
}}

[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
[data-testid="stSidebar"] div[data-testid="stExpander"] summary span,
[data-testid="stSidebar"] div[data-testid="stExpander"] summary p {{
    color: var(--text-sidebar) !important;
}}

.footer {{
    text-align: center;
    color: var(--text-muted);
    font-size: 0.85rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid var(--border);
    margin-top: 1.5rem;
    line-height: 1.55;
}}

.theme-hint {{
    font-size: 0.78rem;
    color: #94a3b8 !important;
    margin-top: 0.35rem;
}}

@keyframes fadeSlide {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes resultPop {{
    from {{ opacity: 0; transform: translateY(14px) scale(0.98); }}
    to {{ opacity: 1; transform: translateY(0) scale(1); }}
}}

/* Inputs & containers — better dark-mode contrast */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
.stTextInput input,
.stNumberInput input {{
    background-color: var(--bg-elevated) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}}

.stSelectbox label,
.stTextInput label,
.stNumberInput label,
.stMarkdown p,
.stCaption,
[data-testid="stWidgetLabel"] p {{
    color: var(--text) !important;
}}

[data-testid="stCaptionContainer"] p {{
    color: var(--text-muted) !important;
}}

div[data-testid="stAlert"] {{
    border-radius: 12px;
}}

[data-testid="stDeckGlJsonChart"],
[data-testid="stMap"] {{
    border-radius: 14px;
    overflow: hidden;
}}

@media (max-width: 768px) {{
    .hero {{ padding: 1.35rem 1.2rem; }}
    .hero h1 {{ font-size: 1.4rem; }}
}}
</style>
"""


@st.cache_resource
def load_assets() -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    bundle = joblib.load(BUNDLE_PATH)
    return config, bundle


def get_theme() -> str:
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "light"
    return st.session_state.ui_theme


def field_label(config: dict[str, Any], feature: str) -> str:
    return config.get("labels", {}).get(feature, feature)


def sync_location(
    config: dict[str, Any],
    reset_quarter: bool = False,
    *,
    clear_prediction: bool = False,
) -> None:
    commune = st.session_state.get("field_Commune")
    locality = config["locality"].get(commune, {})
    quartiers = list(locality.get("quartiers", {}).keys())
    current_quarter = st.session_state.get("field_Quartier")

    if reset_quarter or current_quarter not in quartiers:
        st.session_state.field_Quartier = quartiers[0] if quartiers else None

    quartier = st.session_state.get("field_Quartier")
    quarter_data = locality.get("quartiers", {}).get(quartier, {})
    st.session_state.field_latitude = float(
        quarter_data.get("latitude", locality.get("latitude", -4.325))
    )
    st.session_state.field_longitude = float(
        quarter_data.get("longitude", locality.get("longitude", 15.322))
    )
    st.session_state.zone_display = locality.get(
        "zone", "Zone de résidence non classée"
    )
    if clear_prediction:
        st.session_state.prediction_result = None


def commune_changed(config: dict[str, Any]) -> None:
    sync_location(config, reset_quarter=True, clear_prediction=True)


def quarter_changed(config: dict[str, Any]) -> None:
    sync_location(config, reset_quarter=False, clear_prediction=True)


def initialize_state(config: dict[str, Any]) -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "light"

    for feature, value in config["defaults"].items():
        key = f"field_{feature}"
        if feature == "_Coordonnées GPS du ménage_latitude":
            continue
        if feature == "_Coordonnées GPS du ménage_longitude":
            continue
        if key not in st.session_state:
            st.session_state[key] = value

    if "field_latitude" not in st.session_state:
        st.session_state.field_latitude = 0.0
    if "field_longitude" not in st.session_state:
        st.session_state.field_longitude = 0.0
    if "zone_display" not in st.session_state:
        st.session_state.zone_display = ""

    sync_location(config, reset_quarter=False)


def load_scenario(config: dict[str, Any], scenario_name: str) -> None:
    scenario = config["scenarios"][scenario_name]["inputs"]
    for feature, value in scenario.items():
        if feature == "_Coordonnées GPS du ménage_latitude":
            st.session_state.field_latitude = float(value)
        elif feature == "_Coordonnées GPS du ménage_longitude":
            st.session_state.field_longitude = float(value)
        else:
            st.session_state[f"field_{feature}"] = value
    sync_location(config, reset_quarter=False)
    st.session_state.prediction_result = None


def select_input(
    config: dict[str, Any],
    feature: str,
    *,
    disabled: bool = False,
    help_text: str | None = None,
) -> str | None:
    options = config["categories"].get(feature, [])
    key = f"field_{feature}"
    current = st.session_state.get(key)

    if current not in options and options:
        st.session_state[key] = options[0]

    return st.selectbox(
        field_label(config, feature),
        options=options,
        key=key,
        disabled=disabled,
        help=help_text,
    )


def render_location(config: dict[str, Any]) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.selectbox(
            "Commune",
            options=list(config["locality"].keys()),
            key="field_Commune",
            on_change=commune_changed,
            args=(config,),
            help=(
                "La commune détermine la liste des quartiers et les coordonnées "
                "de référence utilisées par le modèle."
            ),
        )

    commune = st.session_state.field_Commune
    quarters = list(config["locality"][commune]["quartiers"].keys())
    if st.session_state.get("field_Quartier") not in quarters:
        st.session_state.field_Quartier = quarters[0]

    with col2:
        st.selectbox(
            "Quartier",
            options=quarters,
            key="field_Quartier",
            on_change=quarter_changed,
            args=(config,),
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input(
            "Zone de résidence (information)",
            value=st.session_state.zone_display,
            disabled=True,
        )
    with c2:
        st.number_input(
            "Latitude de référence",
            value=float(st.session_state.field_latitude),
            format="%.7f",
            disabled=True,
        )
    with c3:
        st.number_input(
            "Longitude de référence",
            value=float(st.session_state.field_longitude),
            format="%.7f",
            disabled=True,
        )

    st.map(
        pd.DataFrame(
            {
                "lat": [st.session_state.field_latitude],
                "lon": [st.session_state.field_longitude],
            }
        ),
        zoom=11,
        use_container_width=True,
    )
    st.caption(
        "Les coordonnées correspondent à la médiane des ménages enquêtés "
        "dans le quartier sélectionné. Elles servent de référence prédictive "
        "et ne constituent pas l’adresse exacte du ménage."
    )


def render_feature_columns(
    config: dict[str, Any],
    features: list[str],
    columns: int = 2,
) -> None:
    cols = st.columns(columns)
    for index, feature in enumerate(features):
        with cols[index % columns]:
            select_input(config, feature)


def render_profile(config: dict[str, Any]) -> None:
    profile_features = config["groups"]["profil"]
    first_features = [
        feature
        for feature in profile_features
        if feature != "Niveau d’instruction du conjoint/de la conjointe"
    ]
    render_feature_columns(config, first_features, columns=2)

    marital = st.session_state.get("field_Situation matrimoniale")
    if marital in ["Marié(e)", "Union libre"]:
        select_input(
            config,
            "Niveau d’instruction du conjoint/de la conjointe",
        )
    else:
        st.session_state[
            "field_Niveau d’instruction du conjoint/de la conjointe"
        ] = "Non applicable"
        st.info(
            "Le niveau d’instruction du conjoint est automatiquement fixé à "
            "« Non applicable » pour cette situation matrimoniale."
        )


def render_economy(config: dict[str, Any]) -> None:
    insurance_feature = (
        "Au moins un membre du ménage bénéficie-t-il d’une assurance "
        "ou mutuelle de santé ?"
    )
    dependent = set(config["conditional_fields"]["insurance_fields"])
    features = [
        feature
        for feature in config["groups"]["economie"]
        if feature not in dependent
    ]
    render_feature_columns(config, features, columns=2)

    if st.session_state.get(f"field_{insurance_feature}") == "Oui":
        st.markdown("#### Précisions sur la couverture")
        render_feature_columns(
            config,
            config["conditional_fields"]["insurance_fields"],
            columns=2,
        )
    else:
        for feature in dependent:
            st.session_state[f"field_{feature}"] = None
        st.caption(
            "Les questions relatives au type et à l’effet de la couverture "
            "sont masquées lorsque le ménage n’est pas assuré."
        )


def render_environment(config: dict[str, Any]) -> None:
    render_feature_columns(
        config,
        config["groups"]["environnement"],
        columns=2,
    )


def render_supply(config: dict[str, Any]) -> None:
    near_feature = (
        "Existe-t-il une structure de santé formelle proche du domicile ?"
    )
    dependent = set(config["conditional_fields"]["near_structure_fields"])
    general_features = [
        feature
        for feature in config["groups"]["offre"]
        if feature not in dependent and feature != near_feature
    ]

    select_input(config, near_feature)

    if st.session_state.get(f"field_{near_feature}") == "Oui":
        st.markdown("#### Caractéristiques de la structure proche")
        render_feature_columns(
            config,
            config["conditional_fields"]["near_structure_fields"],
            columns=2,
        )
    else:
        for feature in dependent:
            st.session_state[f"field_{feature}"] = None
        st.caption(
            "Les caractéristiques détaillées de la structure sont masquées "
            "lorsque la proximité n’est pas confirmée."
        )

    st.markdown("#### Offre sanitaire du quartier")
    render_feature_columns(config, general_features, columns=2)


def render_need(config: dict[str, Any]) -> None:
    render_feature_columns(
        config,
        config["groups"]["besoin"],
        columns=2,
    )


def render_perception(config: dict[str, Any]) -> None:
    render_feature_columns(
        config,
        config["groups"]["perception"],
        columns=2,
    )


def collect_inputs(config: dict[str, Any]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for feature in config["feature_order"]:
        if feature == "_Coordonnées GPS du ménage_latitude":
            inputs[feature] = float(st.session_state.field_latitude)
        elif feature == "_Coordonnées GPS du ménage_longitude":
            inputs[feature] = float(st.session_state.field_longitude)
        else:
            value = st.session_state.get(f"field_{feature}")
            inputs[feature] = np.nan if value is None else value
    return inputs


def raw_variable_from_encoded(
    encoded_name: str,
    raw_features: list[str],
) -> str:
    core = encoded_name.split("__", 1)[1] if "__" in encoded_name else encoded_name
    if core in raw_features:
        return core

    for raw in sorted(raw_features, key=len, reverse=True):
        if core.startswith(raw + "_"):
            return raw
    return core


@st.cache_resource
def build_explainer(_model: Any) -> Any:
    return shap.TreeExplainer(_model)


def explain_prediction(
    bundle: dict[str, Any],
    row: pd.DataFrame,
) -> pd.DataFrame:
    preprocessor = bundle["preprocessor"]
    transformed = preprocessor.transform(row)
    if hasattr(transformed, "toarray"):
        transformed_dense = transformed.toarray()
    else:
        transformed_dense = np.asarray(transformed)

    explainer = build_explainer(bundle["xgb_model"])
    shap_values = np.asarray(explainer.shap_values(transformed_dense))
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    values = shap_values[0]

    encoded_names = [
        str(name)
        for name in preprocessor.get_feature_names_out()
    ]
    grouped: dict[str, float] = {}
    for encoded_name, contribution in zip(encoded_names, values):
        raw = raw_variable_from_encoded(
            encoded_name,
            bundle["feature_order"],
        )
        grouped[raw] = grouped.get(raw, 0.0) + float(contribution)

    explanation = pd.DataFrame(
        {
            "Variable": list(grouped.keys()),
            "Contribution": list(grouped.values()),
        }
    )
    explanation["Importance"] = explanation["Contribution"].abs()
    explanation["Sens"] = np.where(
        explanation["Contribution"] >= 0,
        "Augmente la probabilité prédite de recours formel",
        "Réduit la probabilité prédite de recours formel",
    )
    return explanation.sort_values(
        "Importance", ascending=False
    ).head(10)


def probability_figure(
    p_access: float,
    threshold: float,
    theme: str,
) -> go.Figure:
    palette = THEME_PALETTES[theme]
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_access * 100,
            number={
                "suffix": " %",
                "font": {"size": 42, "color": palette["plot_font"]},
            },
            title={
                "text": "Probabilité prédite de recours formel",
                "font": {"size": 15, "color": palette["text_muted"]},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": palette["text_muted"],
                    "tickfont": {"color": palette["text_muted"]},
                },
                "bar": {"color": palette["accent"]},
                "bgcolor": palette["bg_muted"],
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, threshold * 100],
                        "color": palette["gauge_low"],
                    },
                    {
                        "range": [threshold * 100, 100],
                        "color": palette["gauge_high"],
                    },
                ],
                "threshold": {
                    "line": {"color": palette["vulnerable"], "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    figure.update_layout(
        height=330,
        margin=dict(l=30, r=30, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": palette["plot_font"]},
    )
    return figure


def shap_figure(explanation: pd.DataFrame, theme: str) -> go.Figure:
    palette = THEME_PALETTES[theme]
    plot_data = explanation.sort_values("Contribution")
    colors = [
        palette["chart_pos"] if value >= 0 else palette["chart_neg"]
        for value in plot_data["Contribution"]
    ]
    figure = go.Figure(
        go.Bar(
            x=plot_data["Contribution"],
            y=plot_data["Variable"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Contribution SHAP: %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title={
            "text": "Principales contributions au résultat individuel",
            "font": {"size": 15, "color": palette["plot_font"]},
        },
        xaxis_title="Contribution SHAP",
        yaxis_title="",
        height=470,
        margin=dict(l=20, r=20, t=55, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": palette["plot_font"]},
        xaxis={
            "gridcolor": palette["plot_grid"],
            "zerolinecolor": palette["plot_grid"],
            "tickfont": {"color": palette["text_muted"]},
            "title_font": {"color": palette["text_muted"]},
        },
        yaxis={
            "tickfont": {"color": palette["text_muted"], "size": 11},
        },
    )
    return figure


def vulnerability_level(p_access: float) -> str:
    if p_access < 0.30:
        return "Vulnérabilité élevée"
    if p_access < 0.50:
        return "Vulnérabilité modérée"
    if p_access < 0.70:
        return "Profil favorable modéré"
    return "Profil favorable élevé"


def run_prediction(
    config: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    inputs = collect_inputs(config)
    row = pd.DataFrame(
        [[inputs[feature] for feature in bundle["feature_order"]]],
        columns=bundle["feature_order"],
    )

    transformed = bundle["preprocessor"].transform(row)
    p_xgb = float(
        bundle["xgb_model"].predict_proba(transformed)[0, 1]
    )
    p_rf = float(
        bundle["rf_model"].predict_proba(transformed)[0, 1]
    )
    threshold = float(config["threshold"])
    predicted_class = int(p_xgb >= threshold)
    profile = (
        "Ménage favorable"
        if predicted_class == 1
        else "Ménage vulnérable"
    )

    try:
        explanation = explain_prediction(bundle, row)
        explanation_error = None
    except Exception as exc:
        explanation = None
        explanation_error = str(exc)

    return {
        "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "inputs": inputs,
        "p_access": p_xgb,
        "p_nonaccess": 1.0 - p_xgb,
        "p_rf": p_rf,
        "predicted_class": predicted_class,
        "profile": profile,
        "level": vulnerability_level(p_xgb),
        "agreement": (p_xgb >= threshold) == (p_rf >= threshold),
        "explanation": explanation,
        "explanation_error": explanation_error,
    }


def render_result(
    config: dict[str, Any],
    bundle: dict[str, Any],
    result: dict[str, Any],
    theme: str,
) -> None:
    favorable = result["predicted_class"] == 1
    css_class = "result-favorable" if favorable else "result-vulnerable"
    fill_class = "good" if favorable else "risk"
    pct = result["p_access"] * 100

    st.markdown(
        f"""
        <div class="result-banner {css_class}">
            <h3>{result["profile"]}</h3>
            <div class="level">{result["level"]}</div>
            <p>
                Probabilité prédite de recours formel :
                <strong style="color:var(--text)">{result["p_access"]:.1%}</strong>
                — seuil officiel à {config["threshold"]:.0%}.
            </p>
            <div class="progress-track">
                <div class="progress-fill {fill_class}" style="width:{pct:.1f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recours formel", f"{result['p_access']:.1%}")
    m2.metric("Non-recours", f"{result['p_nonaccess']:.1%}")
    m3.metric("Seuil officiel", f"{config['threshold']:.0%}")
    m4.metric(
        "Accord XGBoost / RF",
        "Oui" if result["agreement"] else "Non",
    )

    left, right = st.columns([1.0, 1.15], gap="large")
    with left:
        st.plotly_chart(
            probability_figure(
                result["p_access"],
                config["threshold"],
                theme,
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Le classement officiel repose sur le XGBoost. Le Random Forest "
            "est utilisé comme contrôle complémentaire de robustesse."
        )
        st.info(f"Probabilité Random Forest : {result['p_rf']:.1%}")
        if not result["agreement"]:
            st.warning(
                "Les deux modèles aboutissent à des classes différentes. "
                "Le résultat doit être interprété avec une prudence renforcée."
            )

    with right:
        if result["explanation"] is not None:
            explanation = result["explanation"].copy()
            explanation["Variable"] = explanation["Variable"].map(
                lambda value: field_label(config, value)
            )
            st.plotly_chart(
                shap_figure(explanation, theme),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.warning(
                "L’explication SHAP n’a pas pu être calculée : "
                + str(result["explanation_error"])
            )

    with st.expander("Interprétation détaillée des facteurs"):
        if result["explanation"] is not None:
            table = result["explanation"].copy()
            table["Variable"] = table["Variable"].map(
                lambda value: field_label(config, value)
            )
            table["Contribution"] = table["Contribution"].round(4)
            st.dataframe(
                table[["Variable", "Contribution", "Sens"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Une contribution SHAP décrit la manière dont la variable "
                "oriente la prédiction du modèle. Elle ne prouve pas une "
                "relation causale."
            )

    report = {
        "date_prediction": result["timestamp"],
        "modele_principal": bundle["model_name"],
        "modele_secondaire": bundle["secondary_model_name"],
        "seuil": config["threshold"],
        "probabilite_recours_formel": result["p_access"],
        "probabilite_non_recours": result["p_nonaccess"],
        "probabilite_random_forest": result["p_rf"],
        "classe_predite": (
            "Recours formel probable"
            if result["predicted_class"] == 1
            else "Non-recours probable"
        ),
        "profil_menage": result["profile"],
        "niveau": result["level"],
        "accord_modeles": result["agreement"],
        "entrees": result["inputs"],
    }
    st.download_button(
        "Télécharger le rapport de prédiction (JSON)",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="rapport_prediction_recours_soins.json",
        mime="application/json",
        use_container_width=True,
    )


def render_sidebar(config: dict[str, Any]) -> str:
    theme = get_theme()

    st.markdown(
        """
        <div class="brand-mark">
            <div class="logo">✚</div>
            <div class="brand-text">
                <h2>Accès aux soins</h2>
                <p>Kinshasa · recours formel</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    theme_label = "Sombre" if theme == "dark" else "Clair"
    toggle_label = "Passer en mode sombre" if theme == "light" else "Passer en mode clair"
    st.caption(f"Thème actuel : **{theme_label}**")
    if st.button(toggle_label, use_container_width=True, key="theme_toggle"):
        st.session_state.ui_theme = "dark" if theme == "light" else "light"
        st.rerun()

    st.markdown("---")
    st.markdown("##### Performance du modèle")
    metrics = config["model_metrics"]["xgboost"]
    c1, c2 = st.columns(2)
    c1.metric("AUC-ROC", f"{metrics['auc_test']:.3f}")
    c2.metric("Exactitude", f"{metrics['accuracy_test']:.1%}")
    st.metric("Score de Brier", f"{metrics['brier']:.3f}")

    st.markdown("---")
    st.markdown("##### Scénarios de démonstration")
    for scenario_name in config["scenarios"]:
        if st.button(
            scenario_name,
            key=f"scenario_{scenario_name}",
            use_container_width=True,
        ):
            load_scenario(config, scenario_name)
            st.rerun()

    if st.button("Réinitialiser le formulaire", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("field_") or key in [
                "zone_display",
                "prediction_result",
            ]:
                del st.session_state[key]
        initialize_state(config)
        st.rerun()

    if st.session_state.prediction_history:
        with st.expander("Historique de la session"):
            history = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history, use_container_width=True, hide_index=True)
            st.download_button(
                "Exporter l’historique (CSV)",
                data=history.to_csv(index=False).encode("utf-8-sig"),
                file_name="historique_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    return get_theme()


def main() -> None:
    config, bundle = load_assets()
    initialize_state(config)
    theme = get_theme()
    st.markdown(build_theme_css(theme), unsafe_allow_html=True)

    with st.sidebar:
        theme = render_sidebar(config)

    has_result = st.session_state.prediction_result is not None
    step1 = "active"
    step2 = "active" if has_result else ""
    step3 = "active" if has_result else ""

    st.markdown(
        f"""
        <div class="hero">
            <h1>{config["application_title"]}</h1>
            <p>{config["application_subtitle"]}</p>
        </div>
        <div class="steps">
            <div class="step {step1}"><span class="num">1</span> Renseigner le ménage</div>
            <div class="step {step2}"><span class="num">2</span> Calculer la prédiction</div>
            <div class="step {step3}"><span class="num">3</span> Lire le résultat</div>
        </div>
        <div class="status-row">
            <span class="status-chip">XGBoost principal</span>
            <span class="status-chip">Random Forest de contrôle</span>
            <span class="status-chip">54 prédicteurs</span>
            <span class="status-chip">599 recours · 401 non-recours</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-head">
            <h2>Renseignement du ménage</h2>
            <span class="badge">Étape 1</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(
        [
            "Localisation",
            "Profil",
            "Situation économique",
            "Environnement",
            "Offre et mobilité",
            "Besoin de soins",
            "Perception",
        ]
    )

    with tabs[0]:
        render_location(config)
    with tabs[1]:
        render_profile(config)
    with tabs[2]:
        render_economy(config)
    with tabs[3]:
        render_environment(config)
    with tabs[4]:
        render_supply(config)
    with tabs[5]:
        render_need(config)
    with tabs[6]:
        render_perception(config)

    st.markdown(
        """
        <div class="section-head">
            <h2>Calcul de la prédiction</h2>
            <span class="badge">Étape 2</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Le formulaire utilise uniquement des variables disponibles avant ou "
        "au moment de la décision de recours. Les variables postérieures au "
        "recours ont été exclues pour éviter la fuite de cible."
    )

    if st.button(
        "Calculer la probabilité de recours formel",
        type="primary",
        use_container_width=True,
        key="btn_run_prediction",
    ):
        with st.spinner("Prétraitement des données et calcul de la prédiction…"):
            try:
                result = run_prediction(config, bundle)
                st.session_state.prediction_result = result
                st.session_state.prediction_history.append(
                    {
                        "Date": result["timestamp"],
                        "Commune": result["inputs"]["Commune"],
                        "Quartier": result["inputs"]["Quartier"],
                        "Probabilité de recours": round(result["p_access"], 4),
                        "Profil": result["profile"],
                        "Accord des modèles": (
                            "Oui" if result["agreement"] else "Non"
                        ),
                    }
                )
                st.session_state.prediction_history = (
                    st.session_state.prediction_history[-20:]
                )
            except Exception as exc:
                st.session_state.prediction_result = None
                st.error(f"La prédiction a échoué : {exc}")

    if st.session_state.prediction_result:
        st.markdown(
            """
            <div class="section-head">
                <h2>Résultat de la prédiction</h2>
                <span class="badge">Étape 3</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_result(
            config,
            bundle,
            st.session_state.prediction_result,
            theme,
        )

    st.markdown(
        """
        <div class="footer">
            Outil de recherche développé pour le mémoire consacré à l’analyse
            empirique et à la prédiction du recours formel aux soins des
            ménages de Kinshasa.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
