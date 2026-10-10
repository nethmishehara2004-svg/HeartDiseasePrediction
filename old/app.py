import base64
import io
import os
from datetime import datetime

import joblib
import numpy as np
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD TRAINED MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_artifacts():
    model = load_model("heart_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_artifacts()


# ============================================================
# PAGE STATE (form page -> result page)
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "form"
    st.session_state.result = None


# ============================================================
# APPEARANCE SETTINGS (theme, font size, background photo)
# ============================================================

THEMES = {
    "Dark": {
        "bg": "#0b0e14", "bg_grad": "radial-gradient(circle at 20% 0%, #1a1020 0%, #0b0e14 45%)",
        "overlay": "rgba(8,10,16,0.72)", "card": "rgba(18,22,31,0.93)", "solid": "#12161f",
        "card2": "#1a2030", "border": "#2e3648", "text": "#f1f3f8", "muted": "#aab2c2",
        "accent": "#ff5c5c", "accent_text": "#ff8a8a", "track": "#232838",
        "good": "#4ade9c", "bad": "#ff6b6b", "warn": "#facc15", "side": "#0f131c",
    },
    "Light": {
        "bg": "#f4f6fb", "bg_grad": "linear-gradient(160deg, #fdf2f3 0%, #f4f6fb 50%)",
        "overlay": "rgba(255,255,255,0.68)", "card": "rgba(255,255,255,0.95)", "solid": "#ffffff",
        "card2": "#eef1f7", "border": "#d5dbe8", "text": "#1b2233", "muted": "#5b6578",
        "accent": "#e63946", "accent_text": "#d62839", "track": "#e3e8f2",
        "good": "#15935f", "bad": "#d62839", "warn": "#c79100", "side": "#ffffff",
    },
}
FONT_PX = {"Small": 14, "Medium": 15, "Large": 16}


def load_background(uploaded):
    """Return (mime, base64) for the uploaded photo, or a local background file if present."""
    if uploaded is not None:
        if uploaded.size > 4 * 1024 * 1024:
            st.sidebar.warning(
                "Photo is larger than 4 MB. Please use a smaller image.")
            return None
        return uploaded.type, base64.b64encode(uploaded.getvalue()).decode()
    here = os.path.dirname(os.path.abspath(__file__))
    for name, mime in (("background.jpg", "image/jpeg"), ("background.jpeg", "image/jpeg"), ("background.png", "image/png")):
        # looks next to app.py, wherever Streamlit was started
        path = os.path.join(here, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return mime, base64.b64encode(f.read()).decode()
    return None


with st.sidebar:
    st.markdown("<div class='side-title'>Appearance</div>",
                unsafe_allow_html=True)
    theme_name = st.radio("Theme", ["Dark", "Light"], horizontal=True)

size_name = "Large"   # fixed font size
# background comes from background.jpg / background.png in the project folder
photo = None

t = THEMES[theme_name]
bg = load_background(photo)


# ============================================================
# STYLING
# ============================================================

if bg:
    bg_rule = (
        f".stApp {{background-image: linear-gradient({t['overlay']}, {t['overlay']}), "
        f"url(data:{bg[0]};base64,{bg[1]}); background-size: cover; "
        f"background-position: center; background-attachment: fixed;}}"
    )
else:
    bg_rule = f".stApp {{background: {t['bg_grad']};}}"

st.markdown(
    f"""
    <style>
    :root {{
        --bg:{t['bg']}; --card:{t['card']}; --solid:{t['solid']}; --card2:{t['card2']};
        --border:{t['border']}; --text:{t['text']}; --muted:{t['muted']};
        --accent:{t['accent']}; --accent-text:{t['accent_text']}; --track:{t['track']};
        --good:{t['good']}; --bad:{t['bad']}; --warn:{t['warn']};
    }}
    html {{font-size: {FONT_PX[size_name]}px !important;}}
    {bg_rule}
    section[data-testid="stSidebar"] {{background: {t['side']};}}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    *, *::before, *::after {
        box-sizing: border-box !important;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* ---------- BASE RESPONSIVE ROOT FONT SIZES ---------- */
    html { font-size: 16px !important; }
    @media (max-width: 1024px) {
        html { font-size: 15px !important; }
    }
    @media (max-width: 768px) {
        html { font-size: 14.5px !important; }
    }
    @media (max-width: 480px) {
        html { font-size: 14px !important; }
    }

    /* ---------- GLOBAL CONTAINERS & OVERFLOW PREVENTION ---------- */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }

    .block-container {
        max-width: 1250px;
        width: 100% !important;
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
            padding-bottom: 1.5rem !important;
        }
    }

    #MainMenu, footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}

    /* ---------- TOP BAR & NAVIGATION TOGGLE ---------- */
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 99999 !important;
        pointer-events: none !important;
    }
    header[data-testid="stHeader"] * {
        pointer-events: auto !important;
    }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: transparent !important;
        transition: margin-left 0.3s ease, width 0.3s ease;
    }

    /* Fixed, permanently accessible toggle button (open/close icons) */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }

    [data-testid="stSidebarCollapsedControl"] {
        position: fixed !important;
        top: 0.75rem !important;
        left: 0.75rem !important;
    }

    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapseButton"] button,
    button[data-testid="stSidebarCollapseButton"],
    header[data-testid="stHeader"] button {
        width: 44px !important;
        height: 44px !important;
        min-width: 44px !important;
        min-height: 44px !important;
        background: var(--card) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18) !important;
        cursor: pointer !important;
        transition: background-color 0.2s ease, border-color 0.2s ease, transform 0.15s ease !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="stSidebarCollapseButton"] button:hover,
    header[data-testid="stHeader"] button:hover {
        background: var(--card2) !important;
        border-color: var(--accent) !important;
        transform: scale(1.05);
    }

    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg,
    header[data-testid="stHeader"] button svg {
        fill: var(--text) !important;
        color: var(--text) !important;
        width: 22px !important;
        height: 22px !important;
        visibility: visible !important;
    }

    /* ---------- TEXT & WIDGET LABELS ---------- */
    label[data-testid="stWidgetLabel"] p,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label div,
    div[data-testid="stFileUploader"] label p,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span {
        color: var(--text) !important;
    }
    label[data-testid="stWidgetLabel"] p {
        font-weight: 500;
        font-size: 1rem;
        line-height: 1.35;
        margin-bottom: 0.25rem;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--border) !important;
        z-index: 9999 !important;
    }

    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 360px !important;
        min-width: 320px !important;
    }

    @media (max-width: 768px) {
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 85vw !important;
            min-width: 280px !important;
            max-width: 360px !important;
            box-shadow: 4px 0 25px rgba(0,0,0,0.35) !important;
        }
    }

    section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {
        font-size: 1.05rem;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {
        font-size: 1rem;
    }
    .side-title {
        color: var(--accent-text);
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin: 1.4rem 0 0.6rem 0;
    }
    .side-text {
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.6;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 2rem 1.5rem 2rem 1.5rem;
    }
    @media (max-width: 480px) {
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: 1.2rem 1rem 1.5rem 1rem;
        }
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] {margin: 0.6rem 0 1.6rem 0;}
    section[data-testid="stSidebar"] div[role="radiogroup"] {gap: 1.2rem;}
    section[data-testid="stSidebar"] .side-title:first-child {margin-top: 0;}

    /* File Uploader Dropzone */
    div[data-testid="stFileUploaderDropzone"] {
        background: var(--card2) !important;
        border: 1.5px dashed var(--border) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stFileUploaderDropzone"] button {
        background: var(--solid) !important;
        color: var(--text) !important;
        border: 1.5px solid var(--border) !important;
        min-height: 44px !important;
    }
    div[data-testid="stFileUploaderDropzone"] * {color: var(--text) !important;}

    /* ---------- HERO HEADER ---------- */
    .hero {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.8rem 1.6rem;
        margin-bottom: 1.5rem;
        text-align: center;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .hero-title {
        color: var(--accent) !important;
        font-size: clamp(1.35rem, 4vw + 0.3rem, 2.4rem);
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.25;
    }
    .hero-sub {
        color: var(--muted);
        font-size: clamp(0.85rem, 2vw + 0.3rem, 1.05rem);
        margin-top: 0.4rem;
    }
    .chips {margin-top: 1rem;}
    .chip {
        display: inline-block;
        background: var(--card2);
        border: 1px solid var(--border);
        color: var(--text);
        border-radius: 999px;
        padding: 0.3rem 0.8rem;
        font-size: 0.85rem;
        margin: 0.2rem;
    }

    @media (max-width: 480px) {
        .hero {
            border-radius: 14px;
            padding: 1.2rem 0.8rem;
            margin-bottom: 1rem;
        }
    }

    /* ---------- FORM & COLUMNS RESPONSIVENESS ---------- */
    div[data-testid="stForm"] {
        background: transparent;
        border: none;
        padding: 0;
        width: 100%;
    }

    /* Responsive column stacking on mobile and tablet */
    @media (max-width: 767px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 1rem !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:empty {
            display: none !important;
        }
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card);
        border: 1px solid var(--border) !important;
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        width: 100% !important;
        box-sizing: border-box !important;
    }
    @media (max-width: 480px) {
        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.9rem 0.8rem !important;
            border-radius: 14px;
        }
    }

    .field-group {
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--accent-text);
        margin: 0.2rem 0 0.9rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }

    /* Form Inputs */
    div[data-testid="stNumberInput"], div[data-testid="stSelectbox"] {
        width: 100% !important;
        margin-bottom: 0.4rem;
    }

    div[data-testid="stNumberInput"] input {
        background-color: var(--card2) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text) !important;
        font-weight: 500;
        font-size: 1rem;
        padding: 0.6rem 0.7rem !important;
        min-height: 44px !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(255,92,92,0.18) !important;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        background-color: var(--card2) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: var(--card2) !important;
        border: 1.5px solid var(--border) !important;
        color: var(--text) !important;
        min-width: 40px !important;
        min-height: 40px !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        border-color: var(--accent) !important;
    }

    /* Dropdowns */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div {
        background-color: var(--card2) !important;
        color: var(--text) !important;
    }
    div[data-baseweb="select"] > div {
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
        font-weight: 500;
        font-size: 1rem;
        min-height: 44px !important;
        align-items: center;
    }
    div[data-baseweb="select"] * {color: var(--text) !important;}
    div[data-baseweb="select"] svg {fill: var(--text) !important;}
    div[data-baseweb="select"]:focus-within > div {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(255,92,92,0.18) !important;
    }
    div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"] {
        background-color: var(--solid) !important;
        border-radius: 10px !important;
        max-width: 90vw !important;
    }
    div[data-baseweb="popover"] li, div[data-baseweb="popover"] li * {color: var(--text) !important;}
    div[data-baseweb="popover"] li:hover {background-color: var(--card2) !important;}

    /* ---------- BUTTONS ---------- */
    .stFormSubmitButton > button,
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(120deg, #ff5c5c, #e63946);
        color: white;
        font-weight: 700;
        font-size: clamp(1rem, 2.5vw, 1.15rem);
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 8px 24px rgba(230,57,70,0.35);
        transition: transform 0.15s ease, background-color 0.2s ease;
        margin-top: 0.6rem;
        width: 100% !important;
        min-height: 48px !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stFormSubmitButton > button:hover,
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px);
        color: white;
        border: none;
    }
    .stFormSubmitButton > button p,
    div[data-testid="stButton"] > button p,
    div[data-testid="stDownloadButton"] > button p {
        color: white !important;
        margin: 0 !important;
    }

    /* ---------- RESULT PAGE RESPONSIVENESS ---------- */
    .result-card {
        background: var(--card);
        border: 1.5px solid var(--border);
        border-radius: 18px;
        padding: 1.8rem 1.4rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        margin-top: 0.4rem;
        width: 100%;
        box-sizing: border-box;
    }
    @media (max-width: 480px) {
        .result-card {
            padding: 1.2rem 0.85rem;
            border-radius: 14px;
        }
    }

    .result-card.positive {border-color: var(--bad);}
    .result-card.negative {border-color: var(--good);}
    .result-title {
        font-size: clamp(1.2rem, 3.8vw, 1.6rem);
        font-weight: 800;
        color: var(--text);
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .positive .result-title {color: var(--bad);}
    .negative .result-title {color: var(--good);}

<<<<<<< HEAD
    .gauge {
        width: 13.5rem;
        height: 13.5rem;
        max-width: 65vw;
        max-height: 65vw;
        border-radius: 50%;
        margin: 1.4rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        background: conic-gradient(var(--c) calc(var(--p) * 1%), var(--track) 0);
        aspect-ratio: 1 / 1;
    }
    .gauge-inner {
        width: 10rem;
        height: 10rem;
        max-width: 48vw;
        max-height: 48vw;
        border-radius: 50%;
        background: var(--solid);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        aspect-ratio: 1 / 1;
    }
    .gauge-val {
        font-size: clamp(1.4rem, 4.5vw, 2rem);
        font-weight: 800;
        color: var(--text);
    }
    .gauge-lbl {
        font-size: clamp(0.75rem, 2vw, 0.9rem);
        color: var(--muted);
    }
    .metrics {
        display: flex;
        gap: 0.6rem;
        margin-top: 0.6rem;
        flex-wrap: wrap;
    }
    .metric {
        flex: 1 1 110px;
        min-width: 100px;
        background: var(--card2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.75rem 0.4rem;
    }
    .m-val {
        font-size: clamp(1.1rem, 3.5vw, 1.5rem);
        font-weight: 800;
        color: var(--text);
    }
    .m-lbl {
        font-size: clamp(0.75rem, 2vw, 0.9rem);
        color: var(--muted);
    }
    .note {
        margin-top: 1.1rem;
        font-size: clamp(0.8rem, 2vw, 0.9rem);
        color: var(--muted);
        line-height: 1.4;
    }

    .sum-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
        width: 100%;
        box-sizing: border-box;
    }
    @media (max-width: 480px) {
        .sum-card {
            padding: 1rem 0.9rem;
            border-radius: 14px;
        }
    }

    .sum-title {
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--accent-text);
        margin-bottom: 0.8rem;
    }
    .sum-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.4rem 2rem;
    }
    @media (max-width: 640px) {
        .sum-grid {
            grid-template-columns: 1fr;
            gap: 0.2rem;
        }
    }

    .sum-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        border-bottom: 1px solid var(--border);
        font-size: clamp(0.85rem, 2.5vw, 1rem);
        color: var(--muted);
        gap: 0.5rem;
    }
    .sum-row span {
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .sum-row b {
        color: var(--text);
        text-align: right;
        white-space: nowrap;
=======
    .sum-card {background: var(--card); border: 1px solid var(--border); border-radius: 18px;
        padding: 1.4rem 1.8rem; margin-top: 1.2rem;}
    .sum-title {font-size: 0.9rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
        color: var(--accent-text); margin-bottom: 0.8rem;}
    .sum-grid {display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem 2.5rem;}
    .sum-row {display: flex; justify-content: space-between; padding: 0.45rem 0;
        border-bottom: 1px solid var(--border); font-size: 1rem; color: var(--muted);}
    .sum-row b {color: var(--text);}

    /* ---------- DROPDOWNS: white in Light mode, dark in Dark mode ---------- */
    div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] ~ * {
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] ~ *,
    div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] ~ * * {
        background-color: var(--solid) !important;
        color: var(--text) !important;
    }
    div[data-testid="stSelectbox"] svg {fill: var(--text) !important;}
    ul[role="listbox"], li[role="option"] {
        background-color: var(--solid) !important;
        color: var(--text) !important;
>>>>>>> origin/main
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

with st.sidebar:
    st.markdown(
        "<div class='side-title'>About</div>"
        "<div class='side-text'>An AI-based tool that estimates the likelihood of heart disease "
        "from 13 clinical measurements using a trained neural network.</div>"
        "<div class='side-title'>How to use</div>"
        "<div class='side-text'>1. Fill in both cards.<br>2. Click Predict.<br>3. The result opens on a new page.</div>"
        "<div class='side-title'>Disclaimer</div>"
        "<div class='side-text'>This tool supports screening only. It is not a medical diagnosis. "
        "Always consult a qualified doctor.</div>",
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    "<div class='hero'>"
    "<div class='hero-title'>❤️ Heart Disease Prediction System</div>"
    "<div class='hero-sub'>Enter the 13 patient details to estimate the risk of heart disease</div>"
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# MAPPINGS (must match training encoding exactly)
# ============================================================

CP_MAP = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

FBS_MAP = {"No": 0, "Yes": 1}

RESTECG_MAP = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

EXANG_MAP = {"No": 0, "Yes": 1}

SLOPE_MAP = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

THAL_MAP = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}

LEVEL_COLOUR = {"Low": "good", "Moderate": "warn", "High": "bad"}


# ============================================================
# DOWNLOADABLE REPORT (PDF)
# ============================================================

def build_report_pdf(r):
    """Return the prediction report as PDF bytes."""
    colours = {"Low": "#15935f", "Moderate": "#c79100", "High": "#d62839"}
    colour = colors.HexColor(colours[r["level"]])
    plain_title = r["title"].replace("🔴", "").replace("🟢", "").strip()
    generated = datetime.now().strftime("%d %B %Y, %H:%M")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("t", parent=styles["Title"], textColor=colors.HexColor("#e63946"))
    meta_style = ParagraphStyle("m", parent=styles["Normal"], textColor=colors.HexColor("#5b6578"),
                                alignment=1, spaceAfter=14)
    result_style = ParagraphStyle("r", parent=styles["Heading2"], textColor=colour, alignment=1)
    note_style = ParagraphStyle("n", parent=styles["Normal"], fontSize=9,
                                textColor=colors.HexColor("#5b6578"))

    # headline numbers
    stats = Table(
        [[f"{r['risk_pct']:.1f}%", f"{r['conf']:.1f}%", r["level"]],
         ["Disease probability", "Confidence", "Risk level"]],
        colWidths=[5.5 * cm] * 3,
    )
    stats.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 18),
        ("TEXTCOLOR", (2, 0), (2, 0), colour),
        ("FONTSIZE", (0, 1), (-1, 1), 9),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#5b6578")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef1f7")),
        ("BOX", (0, 0), (-1, -1), 1.5, colour),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    # patient details
    details = Table([[str(k), str(v)] for k, v in r["inputs"].items()],
                    colWidths=[10 * cm, 6.5 * cm])
    details.setStyle(TableStyle([
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#5b6578")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#d5dbe8")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story = [
        Paragraph("Heart Disease Prediction Report", title_style),
        Paragraph(f"Generated on {generated}", meta_style),
        Paragraph(plain_title, result_style),
        Spacer(1, 6),
        stats,
        Spacer(1, 18),
        Paragraph("Patient details used for this prediction", styles["Heading3"]),
        details,
        Spacer(1, 24),
        Paragraph("This report supports screening only and is not a medical diagnosis. "
                  "Please consult a qualified doctor for any medical decision.", note_style),
    ]

    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, title="Heart Disease Prediction Report",
                      leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm).build(story)
    return buffer.getvalue()


# ============================================================
# PAGE 1: INPUT FORM
# ============================================================

if st.session_state.page == "form":

    with st.form("patient_form"):

        col1, col2 = st.columns(2, gap="large")

        with col1:
            with st.container(border=True):
                st.markdown(
                    '<div class="field-group">🧍 Patient Information</div>', unsafe_allow_html=True)
                age = st.number_input("Age", min_value=1,
                                      max_value=120, value=50)
                gender = st.selectbox("Gender", ["Male", "Female"])
                cp = st.selectbox("Chest Pain Type", list(CP_MAP.keys()))
                trestbps = st.number_input(
                    "Resting Blood Pressure", min_value=0, max_value=300, value=120)
                chol = st.number_input(
                    "Cholesterol", min_value=0, max_value=700, value=200)
                fbs = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dl", list(FBS_MAP.keys()))
                restecg = st.selectbox(
                    "Resting ECG Result", list(RESTECG_MAP.keys()))

        with col2:
            with st.container(border=True):
                st.markdown(
                    '<div class="field-group">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
                thalach = st.number_input(
                    "Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
                exang = st.selectbox(
                    "Exercise Induced Angina", list(EXANG_MAP.keys()))
                oldpeak = st.number_input(
                    "Old Peak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                slope = st.selectbox(
                    "Slope of Peak Exercise ST", list(SLOPE_MAP.keys()))
                ca = st.selectbox("Number of Major Vessels",
                                  ["0", "1", "2", "3", "4"])
                thal = st.selectbox("Thalassemia", list(THAL_MAP.keys()))

        submitted = st.form_submit_button(
            "🔍 Predict", use_container_width=True)

    # ------------------------------------------------------------
    # PREDICTION (runs on submit, then moves to the result page)
    # ------------------------------------------------------------

    if submitted:
        try:
            sex = 1 if gender == "Male" else 0

            patient = np.array([[
                float(age),
                sex,
                CP_MAP[cp],
                float(trestbps),
                float(chol),
                FBS_MAP[fbs],
                RESTECG_MAP[restecg],
                float(thalach),
                EXANG_MAP[exang],
                float(oldpeak),
                SLOPE_MAP[slope],
                int(ca),
                THAL_MAP[thal]
            ]])

            patient_scaled = scaler.transform(patient)

            prediction = model.predict(patient_scaled, verbose=0)
            probability = float(prediction[0][0])

            risk_pct = probability * 100

            if risk_pct < 20:
                level = "Low"
            elif risk_pct < 50:
                level = "Moderate"
            else:
                level = "High"

            if probability >= 0.5:
                title, css_class, conf = "🔴 Heart Disease Detected", "positive", probability * 100
            else:
                title, css_class, conf = "🟢 No Heart Disease Detected", "negative", (
                    1 - probability) * 100

            st.session_state.result = {
                "title": title, "css_class": css_class, "risk_pct": risk_pct,
                "conf": conf, "level": level,
                "inputs": {
                    "Age": age, "Gender": gender, "Chest Pain Type": cp,
                    "Resting Blood Pressure": trestbps, "Cholesterol": chol,
                    "Fasting Blood Sugar > 120": fbs, "Resting ECG": restecg,
                    "Max Heart Rate": thalach, "Exercise Angina": exang,
                    "Old Peak": f"{oldpeak:.2f}", "Slope": slope,
                    "Major Vessels": ca, "Thalassemia": thal,
                },
            }
            st.session_state.page = "result"
            st.rerun()

        except Exception as e:
            st.error(f"Something went wrong: {e}")


# ============================================================
# PAGE 2: RESULT
# ============================================================

else:
    r = st.session_state.result
    color = t[LEVEL_COLOUR[r["level"]]]

    _, centre, _ = st.columns([1, 2, 1])

    with centre:
        st.markdown(
            f"<div class='result-card {r['css_class']}'>"
            f"<div class='result-title'>{r['title']}</div>"
            f"<div class='gauge' style='--p:{r['risk_pct']:.1f}; --c:{color};'>"
            f"<div class='gauge-inner'>"
            f"<div class='gauge-val'>{r['risk_pct']:.1f}%</div>"
            f"<div class='gauge-lbl'>Disease probability</div>"
            f"</div></div>"
            f"<div class='metrics'>"
            f"<div class='metric'><div class='m-val'>{r['conf']:.1f}%</div><div class='m-lbl'>Confidence</div></div>"
            f"<div class='metric'><div class='m-val' style='color:{color};'>{r['level']}</div><div class='m-lbl'>Risk level</div></div>"
            f"</div>"
            f"<div class='note'>This tool supports screening only and is not a medical diagnosis.</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    rows = "".join(
        f"<div class='sum-row'><span>{k}</span><b>{v}</b></div>"
        for k, v in r["inputs"].items()
    )
    st.markdown(
        f"<div class='sum-card'><div class='sum-title'>📋 Patient details used for this prediction</div>"
        f"<div class='sum-grid'>{rows}</div></div>",
        unsafe_allow_html=True
    )

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        # optional: the user can download the report if they want it
        st.download_button(
            "📄 Download report (PDF)",
            data=build_report_pdf(r),
            file_name=f"heart_report_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        if st.button("← New prediction", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()