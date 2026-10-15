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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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
    model = load_model(os.path.join(BASE_DIR, "heart_model.keras"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    return model, scaler


model, scaler = load_artifacts()


# ============================================================
# TRANSLATIONS (English text is key; supports Sinhala & Tamil)
# ============================================================

LANGUAGES = {"English": "en", "සිංහල": "si", "தமிழ்": "ta"}

TRANSLATIONS = {
    "si": {
        "Heart Disease Prediction System": "හෘද රෝග පුරෝකථන පද්ධතිය",
        "Enter the 13 patient details to estimate the risk of heart disease":
            "හෘද රෝග අවදානම ඇස්තමේන්තු කිරීමට රෝගියාගේ තොරතුරු 13 ඇතුළත් කරන්න",
        "Appearance": "පෙනුම",
        "Theme": "තේමාව",
        "Dark": "අඳුරු",
        "Light": "ආලෝක",
        "Language": "භාෂාව",
        "About": "පිළිබඳව",
        "How to use": "භාවිතා කරන ආකාරය",
        "Disclaimer": "වියාචනය",
        "ABOUT_TEXT": "සායනික මිනුම් 13ක් පදනම් කරගෙන, පුහුණු කළ ස්නායුක ජාලයක් මගින් හෘද රෝග ඇතිවීමේ සම්භාවිතාව ඇස්තමේන්තු කරන AI මෙවලමකි.",
        "HOWTO_TEXT": "1. කාඩ්පත් දෙකම පුරවන්න.<br>2. පුරෝකථනය කරන්න ඔබන්න.<br>3. ප්‍රතිඵලය නව පිටුවක විවෘත වේ.",
        "DISCLAIMER_TEXT": "මෙම මෙවලම මූලික පරීක්ෂාවක් සඳහා පමණක් උපකාරී වන අතර, එය වෛද්‍ය රෝග විනිශ්චයක් සඳහා ආදේශකයක් නොවේ. සෑම විටම සුදුසුකම් ලත් වෛද්‍යවරයෙකුගෙන් උපදෙස් ලබාගන්න.",
        "This tool supports screening only and is not a medical diagnosis.":
            "මෙම මෙවලම මූලික පරීක්ෂාවක් සඳහා පමණක් උපකාරී වන අතර, එය වෛද්‍ය රෝග විනිශ්චයක් සඳහා ආදේශකයක් නොවේ.",
        "Patient Information": "රෝගී තොරතුරු",
        "Clinical Measurements": "සායනික මිනුම්",
        "Predict": "පුරෝකථනය කරන්න",
        "Clear form": "පෝරමය හිස් කරන්න",
        "New prediction": "නව පුරෝකථනයක්",
        "Download report (PDF)": "වාර්තාව බාගන්න (PDF)",
        "Age": "වයස",
        "Gender": "ස්ත්‍රී පුරුෂ භාවය",
        "Male": "පුරුෂ",
        "Female": "ස්ත්‍රී",
        "Chest Pain Type": "පපුවේ වේදනා වර්ගය",
        "Typical Angina": "සාමාන්‍ය ඇන්ජයිනාව",
        "Atypical Angina": "අසාමාන්‍ය ඇන්ජයිනාව",
        "Non-anginal Pain": "ඇන්ජයිනා නොවන වේදනාව",
        "Asymptomatic": "රෝග ලක්ෂණ නොමැති",
        "Resting Blood Pressure": "විවේක රුධිර පීඩනය",
        "Cholesterol": "කොලෙස්ටරෝල්",
        "Fasting Blood Sugar > 120 mg/dl": "නිරාහාර රුධිර සීනි > 120 mg/dL",
        "Yes": "ඔව්",
        "No": "නැත",
        "Resting ECG Result": "විවේක ECG ප්‍රතිඵලය",
        "Normal": "සාමාන්‍ය",
        "ST-T Abnormality": "ST-T අසාමාන්‍යතාව",
        "Left Ventricular Hypertrophy": "වම් හෘද කෝෂිකාවේ අධිවර්ධනය",
        "Maximum Heart Rate Achieved": "ළඟා වූ උපරිම හෘද ස්පන්දන වේගය",
        "Exercise Induced Angina": "ව්‍යායාමයෙන් ඇති වන ඇන්ජයිනාව",
        "Old Peak (ST Depression)": "ST අවපීඩන අගය (ඕල්ඩ් පීක්)",
        "Slope of Peak Exercise ST": "උපරිම ව්‍යායාමයේදී ST කොටසේ බෑවුම",
        "Upsloping": "ඉහළට බෑවුම්",
        "Flat": "සමතලා",
        "Downsloping": "පහළට බෑවුම්",
        "Number of Major Vessels": "ප්‍රධාන රුධිර නාල ගණන",
        "Thalassemia": "තැලසීමියාව",
        "Fixed Defect": "ස්ථීර දෝෂය",
        "Reversible Defect": "ප්‍රතිවර්ත්‍ය දෝෂය",
        "Heart Disease Detected": "හෘද රෝග හඳුනාගෙන ඇත",
        "No Heart Disease Detected": "හෘද රෝග හඳුනාගෙන නොමැත",
        "Low risk of heart disease": "හෘද රෝග අවදානම අඩුයි",
        "Moderate risk of heart disease": "හෘද රෝග අවදානම මධ්‍යමයි",
        "High risk of heart disease": "හෘද රෝග අවදානම ඉහළයි",
        "Disease probability": "රෝග සම්භාවිතාව",
        "Confidence": "ආකෘතියේ නිශ්චිතභාවය",
        "Model certainty": "ආකෘතියේ නිශ්චිතභාවය",
        "Risk level": "අවදානම් මට්ටම",
        "Low": "අඩු",
        "Moderate": "මධ්‍යම",
        "High": "ඉහළ",
        "Patient details used for this prediction": "මෙම පුරෝකථනයට භාවිතා කළ රෝගී තොරතුරු",
        "Something went wrong": "යම් දෝෂයක් සිදු විය",
        "Heart Disease Prediction Report": "හෘද රෝග පුරෝකථන වාර්තාව",
        "Generated on": "සකස් කළ දිනය",
        "Please consult a qualified doctor for any medical decision.":
            "වෛද්‍ය තීරණ සඳහා සුදුසුකම් ලත් වෛද්‍යවරයෙකුගෙන් උපදෙස් ලබාගන්න.",
    },
    "ta": {
        "Heart Disease Prediction System": "இதய நோய் முன்கணிப்பு அமைப்பு",
        "Enter the 13 patient details to estimate the risk of heart disease":
            "இதய நோய் ஆபத்தை மதிப்பிட நோயாளியின் 13 விவரங்களை உள்ளிடவும்",
        "Appearance": "தோற்றம்",
        "Theme": "தீம்",
        "Dark": "இருண்ட",
        "Light": "ஒளி",
        "Language": "மொழி",
        "About": "பற்றி",
        "How to use": "பயன்படுத்தும் முறை",
        "Disclaimer": "மறுப்பு அறிக்கை",
        "ABOUT_TEXT": "13 மருத்துவ அளவீடுகளின் அடிப்படையில், பயிற்சி பெற்ற நரம்பியல் வலையமைப்பைப் பயன்படுத்தி இதய நோய்க்கான வாய்ப்பை மதிப்பிடும் AI கருவி.",
        "HOWTO_TEXT": "1. இரு அட்டைகளையும் நிரப்பவும்.<br>2. கணிக்கவும் என்பதை அழுத்தவும்.<br>3. முடிவு புதிய பக்கத்தில் திறக்கும்.",
        "DISCLAIMER_TEXT": "இந்தக் கருவி ஆரம்பநிலைப் பரிசோதனைக்கு மட்டுமே உதவுகிறது; இது மருத்துவ நோயறிதலுக்கு மாற்றாகாது. எப்போதும் தகுதிவாய்ந்த மருத்துவரின் ஆலோசனையைப் பெறவும்.",
        "This tool supports screening only and is not a medical diagnosis.":
            "இந்தக் கருவி ஆரம்பநிலைப் பரிசோதனைக்கு மட்டுமே உதவுகிறது; இது மருத்துவ நோயறிதலுக்கு மாற்றாகாது.",
        "Patient Information": "நோயாளி விவரங்கள்",
        "Clinical Measurements": "மருத்துவ அளவீடுகள்",
        "Predict": "கணிக்கவும்",
        "Clear form": "படிவத்தை அழி",
        "New prediction": "புதிய கணிப்பு",
        "Download report (PDF)": "அறிக்கையைப் பதிவிறக்கு (PDF)",
        "Age": "வயது",
        "Gender": "பாலினம்",
        "Male": "ஆண்",
        "Female": "பெண்",
        "Chest Pain Type": "நெஞ்சு வலி வகை",
        "Typical Angina": "வழக்கமான ஆஞ்சினா",
        "Atypical Angina": "வழக்கத்திற்கு மாறான ஆஞ்சினா",
        "Non-anginal Pain": "ஆஞ்சினா அல்லாத வலி",
        "Asymptomatic": "அறிகுறிகள் இல்லாத நிலை",
        "Resting Blood Pressure": "ஓய்வு நிலை இரத்த அழுத்தம்",
        "Cholesterol": "கொலஸ்ட்ரால்",
        "Fasting Blood Sugar > 120 mg/dl": "உண்ணாவிரத இரத்தச் சர்க்கரை > 120 mg/dL",
        "Yes": "ஆம்",
        "No": "இல்லை",
        "Resting ECG Result": "ஓய்வு நிலை ECG முடிவு",
        "Normal": "இயல்பானது",
        "ST-T Abnormality": "ST-T அசாதாரணம்",
        "Left Ventricular Hypertrophy": "இடது இதயக் கீழறை தசை தடிப்பு",
        "Maximum Heart Rate Achieved": "அடைந்த அதிகபட்ச இதயத் துடிப்பு",
        "Exercise Induced Angina": "உடற்பயிற்சியால் ஏற்படும் ஆஞ்சினா",
        "Old Peak (ST Depression)": "ST தாழ்வு அளவு (ஓல்ட் பீக்)",
        "Slope of Peak Exercise ST": "உச்ச உடற்பயிற்சியின் போது ST பகுதியின் சரிவு",
        "Upsloping": "மேல்நோக்கிய சரிவு",
        "Flat": "தட்டையான சரிவு",
        "Downsloping": "கீழ்நோக்கிய சரிவு",
        "Number of Major Vessels": "முக்கிய இரத்த நாளங்களின் எண்ணிக்கை",
        "Thalassemia": "தலசீமியா",
        "Fixed Defect": "நிலையான குறைபாடு",
        "Reversible Defect": "மீளக்கூடிய குறைபாடு",
        "Heart Disease Detected": "இதய நோய் கண்டறியப்பட்டது",
        "No Heart Disease Detected": "இதய நோய் கண்டறியப்படவில்லை",
        "Low risk of heart disease": "இதய நோய் ஆபத்து குறைவு",
        "Moderate risk of heart disease": "இதய நோய் ஆபத்து மிதமானது",
        "High risk of heart disease": "இதய நோய் ஆபத்து அதிகம்",
        "Disease probability": "நோய் நிகழ்தகவு",
        "Confidence": "மாதிரியின் உறுதிநிலை",
        "Model certainty": "மாதிரியின் உறுதிநிலை",
        "Risk level": "ஆபத்து நிலை",
        "Low": "குறைவு",
        "Moderate": "மிதமான",
        "High": "அதிகம்",
        "Patient details used for this prediction": "இந்தக் கணிப்புக்குப் பயன்படுத்தப்பட்ட நோயாளி விவரங்கள்",
        "Something went wrong": "ஏதோ தவறு நடந்துவிட்டது",
        "Heart Disease Prediction Report": "இதய நோய் முன்கணிப்பு அறிக்கை",
        "Generated on": "உருவாக்கிய தேதி",
        "Please consult a qualified doctor for any medical decision.":
            "எந்த மருத்துவ முடிவுக்கும் தகுதிவாய்ந்த மருத்துவரை அணுகவும்.",
    },
}

EN_TEXT = {
    "ABOUT_TEXT": "An AI-based tool that estimates the likelihood of heart disease "
                  "from 13 clinical measurements using a trained neural network.",
    "HOWTO_TEXT": "1. Fill in both cards.<br>2. Click Predict.<br>3. The result opens on a new page.",
    "DISCLAIMER_TEXT": "This tool supports screening only. It is not a medical diagnosis. "
                       "Always consult a qualified doctor.",
}


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
        "bg": "#071426", "bg_grad": "radial-gradient(circle at 20% 0%, #0d1e38 0%, #071426 45%)",
        "overlay": "rgba(7,20,38,0.72)", "card": "#132238", "solid": "#132238",
        "card2": "#1E293B", "border": "#334155", "heading": "#F8FAFC",
        "text": "#CBD5E1", "muted": "#94A3B8",
        "input_bg": "#1E293B", "input_text": "#F8FAFC",
        "menu_hover": "#2B394E", "menu_hover_text": "#60A5FA",
        "accent": "#60A5FA", "accent_text": "#60A5FA", "track": "#1E293B",
        "btn_bg": "#2563EB", "btn_text": "#FFFFFF",
        "good": "#34D399", "bad": "#F87171", "warn": "#FBBF24", "side": "#132238",
    },
    "Light": {
        "bg": "#F3F8FF", "bg_grad": "linear-gradient(160deg, #e8f0fe 0%, #F3F8FF 50%)",
        "overlay": "rgba(243,248,255,0.70)", "card": "#FFFFFF", "solid": "#FFFFFF",
        "card2": "#F3F8FF", "border": "#CBD5E1", "heading": "#12304A",
        "text": "#475569", "muted": "#64748B",
        "input_bg": "#FFFFFF", "input_text": "#12304A",
        "menu_hover": "#EBF3FE", "menu_hover_text": "#2563EB",
        "accent": "#2563EB", "accent_text": "#2563EB", "track": "#E2E8F0",
        "btn_bg": "#2563EB", "btn_text": "#FFFFFF",
        "good": "#15803D", "bad": "#DC2626", "warn": "#D97706", "side": "#FFFFFF",
    },
}
FONT_PX = {"Small": 14, "Medium": 18, "Large": 22}


@st.cache_data
def _read_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_background(theme_name="Dark", uploaded=None):
    """Return (mime, base64) for the uploaded photo, or a local theme-specific background file."""
    if uploaded is not None:
        if uploaded.size > 4 * 1024 * 1024:
            st.sidebar.warning(
                "Photo is larger than 4 MB. Please use a smaller image.")
            return None
        return uploaded.type, base64.b64encode(uploaded.getvalue()).decode()

    here = BASE_DIR
    prefix = "dark_bg" if theme_name == "Dark" else "light_bg"
    candidates = [
        (f"{prefix}.png", "image/png"),
        (f"{prefix}.jpg", "image/jpeg"),
        (f"{prefix}.jpeg", "image/jpeg"),
        (f"{prefix}.webp", "image/webp"),
        ("background.jpg", "image/jpeg"),
        ("background.jpeg", "image/jpeg"),
        ("background.png", "image/png"),
    ]

    for name, mime in candidates:
        path = os.path.join(here, name)
        if os.path.exists(path):
            return mime, _read_image_b64(path)
    return None


# ============================================================
# TOP HEADER BAR: LANGUAGE & THEME
# ============================================================

top_lang, top_theme = st.columns([3, 2], gap="large")

with top_lang:
    lang_name = st.radio("Language / භාෂාව / மொழி", list(LANGUAGES.keys()),
                         horizontal=True, key="lang_name")

lang = LANGUAGES[lang_name]


def tr(text):
    """Translate into chosen language or fallback to English."""
    if lang == "en":
        return EN_TEXT.get(text, text)
    return TRANSLATIONS.get(lang, {}).get(text, EN_TEXT.get(text, text))


with top_theme:
    theme_name = st.radio(tr("Theme"), ["Dark", "Light"], horizontal=True,
                          format_func=tr, key="theme_name")

size_name = "Large"   # fixed font size
photo = None

t = THEMES[theme_name]
bg = load_background(theme_name, photo)


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
        --border:{t['border']}; --heading:{t['heading']}; --text:{t['text']}; --muted:{t['muted']};
        --input-bg:{t['input_bg']}; --input-text:{t['input_text']};
        --menu-hover:{t['menu_hover']}; --menu-hover-text:{t['menu_hover_text']};
        --accent:{t['accent']}; --accent-text:{t['accent_text']};
        --btn-bg:{t['btn_bg']}; --btn-text:{t['btn_text']};
        --track:{t['track']}; --good:{t['good']}; --bad:{t['bad']}; --warn:{t['warn']};
    }}
    html {{font-size: {FONT_PX[size_name]}px !important;}}
    {bg_rule}
    section[data-testid="stSidebar"] {{background: {t['side']} !important;}}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Sinhala:wght@400;500;600;700&family=Noto+Sans+Tamil:wght@400;500;600;700&display=swap');

    *, *::before, *::after {
        box-sizing: border-box !important;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans Sinhala', 'Noto Sans Tamil', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* ---------- BASE RESPONSIVE ROOT FONT SIZES (big on desktop) ---------- */
    html { font-size: 22px !important; }
    @media (max-width: 1024px) {
        html { font-size: 20px !important; }
    }
    @media (max-width: 768px) {
        html { font-size: 18px !important; }
    }
    @media (max-width: 480px) {
        html { font-size: 16px !important; }
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
        width: 440px !important;
        min-width: 320px !important;
        max-width: 45vw !important;
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
        color: var(--heading) !important;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin: 1.4rem 0 0.6rem 0;
    }
    .side-text {
        color: var(--text);
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
        color: var(--heading) !important;
        font-size: clamp(1.35rem, 4vw + 0.3rem, 2.4rem);
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.25;
    }
    .hero-sub {
        color: var(--text);
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
        color: var(--heading);
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
        background-color: var(--input-bg) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--input-text) !important;
        font-weight: 500;
        font-size: 1rem;
        padding: 0.6rem 0.7rem !important;
        min-height: 44px !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.2) !important;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        background-color: var(--input-bg) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stNumberInput"] button {
        background-color: var(--input-bg) !important;
        border: 1.5px solid var(--border) !important;
        color: var(--input-text) !important;
        min-width: 40px !important;
        min-height: 40px !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        border-color: var(--accent) !important;
    }

    /* ============================================================
       SELECTBOX INPUT BOX (LIGHT & DARK MODE)
       ============================================================ */
    div[data-testid="stSelectbox"] {
        width: 100% !important;
        margin-bottom: 0.4rem;
    }

    /* Outer input box container */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stSelectbox"] div[role="group"],
    div[data-testid="stSelectbox"] [class*="react-aria-Group"],
    div[data-testid="stSelectbox"] [class*="e1fp86qc0"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div {
        background-color: var(--input-bg) !important;
        background: var(--input-bg) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--input-text) !important;
    }

    /* Inner input text & placeholder */
    div[data-testid="stSelectbox"] input,
    div[data-testid="stSelectbox"] [class*="e1fp86qc1"],
    div[data-baseweb="select"] input {
        background: transparent !important;
        background-color: transparent !important;
        color: var(--input-text) !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }

    /* Dropdown arrow icon / button */
    div[data-testid="stSelectbox"] button,
    div[data-testid="stSelectbox"] [class*="e1fp86qc2"],
    div[data-testid="stSelectbox"] [class*="e1fp86qc3"],
    div[data-testid="stSelectbox"] svg,
    div[data-baseweb="select"] svg {
        background: transparent !important;
        background-color: transparent !important;
        color: var(--input-text) !important;
        fill: var(--input-text) !important;
    }

    /* Focus state */
    div[data-testid="stSelectbox"] > div:focus-within,
    div[data-testid="stSelectbox"] div[data-focus-within],
    div[data-baseweb="select"]:focus-within > div {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.2) !important;
    }

    /* ============================================================
       DROPDOWN MENU POPUP (LIGHT & DARK MODE)
       ============================================================ */
    [data-testid="stSelectboxVirtualDropdown"],
    [data-testid="stSelectboxVirtualDropdown"] > div,
    [data-testid="stSelectboxVirtualDropdown"] div,
    [data-testid="stSelectboxVirtualDropdown"] ul,
    [class*="e1fp86qc4"],
    [class*="e1fp86qc5"],
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] div,
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] [data-baseweb="menu"],
    div[data-baseweb="popover"] [role="listbox"],
    ul[data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: var(--input-bg) !important;
        background: var(--input-bg) !important;
        color: var(--input-text) !important;
        border-color: var(--border) !important;
    }

    [data-testid="stSelectboxVirtualDropdown"],
    div[data-baseweb="popover"] > div {
        border: 1.5px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15) !important;
        overflow: hidden !important;
        max-width: 90vw !important;
    }

    /* Option Items (Default State) */
    [data-testid="stSelectboxVirtualDropdown"] li,
    [data-testid="stSelectboxVirtualDropdown"] [class*="e1fp86qc7"],
    [data-testid="stSelectboxVirtualDropdown"] span,
    [data-testid="stSelectboxVirtualDropdown"] div,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li > div,
    div[data-baseweb="popover"] li > span,
    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="popover"] [role="option"] > div,
    div[data-baseweb="popover"] [data-baseweb="menu-item"],
    ul[role="listbox"] li,
    li[role="option"] {
        background-color: var(--input-bg) !important;
        background: var(--input-bg) !important;
        color: var(--input-text) !important;
    }

    /* Inner item highlight container */
    [data-testid="stSelectboxVirtualDropdown"] [data-item-hl],
    [data-testid="stSelectboxVirtualDropdown"] [class*="e1fp86qc8"] {
        background-color: transparent !important;
        background: transparent !important;
        color: var(--input-text) !important;
    }

    [data-testid="stSelectboxVirtualDropdown"] * {
        color: var(--input-text) !important;
    }

    /* Option Items (Hover, Focused, Selected State) */
    [data-testid="stSelectboxVirtualDropdown"] li:hover [data-item-hl],
    [data-testid="stSelectboxVirtualDropdown"] li[data-hovered] [data-item-hl],
    [data-testid="stSelectboxVirtualDropdown"] li[data-focused] [data-item-hl],
    [data-testid="stSelectboxVirtualDropdown"] [class*="e1fp86qc7"]:hover [data-item-hl],
    [data-testid="stSelectboxVirtualDropdown"] [class*="e1fp86qc7"][data-hovered] [data-item-hl],
    [data-testid="stSelectboxVirtualDropdown"] [class*="e1fp86qc7"][data-focused] [data-item-hl],
    [data-testid="stSelectboxVirtualDropdown"] [data-item-hl]:hover,
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] li:hover > div,
    div[data-baseweb="popover"] li[aria-selected="true"],
    div[data-baseweb="popover"] li[aria-selected="true"] > div,
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [role="option"]:hover > div,
    div[data-baseweb="popover"] [role="option"][aria-selected="true"],
    div[data-baseweb="popover"] [role="option"][aria-selected="true"] > div,
    ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"],
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: var(--menu-hover) !important;
        background: var(--menu-hover) !important;
        color: var(--menu-hover-text) !important;
    }

    [data-testid="stSelectboxVirtualDropdown"] li:hover *,
    [data-testid="stSelectboxVirtualDropdown"] li[data-hovered] *,
    [data-testid="stSelectboxVirtualDropdown"] li[data-focused] *,
    div[data-baseweb="popover"] li:hover *,
    div[data-baseweb="popover"] li[aria-selected="true"] * {
        color: var(--menu-hover-text) !important;
    }

    /* Custom scrollbar in dropdown menu */
    [data-testid="stSelectboxVirtualDropdown"] ul::-webkit-scrollbar,
    [data-testid="stSelectboxVirtualDropdown"] div::-webkit-scrollbar,
    div[data-baseweb="popover"] ul::-webkit-scrollbar,
    div[data-baseweb="popover"] div::-webkit-scrollbar {
        width: 6px;
    }
    [data-testid="stSelectboxVirtualDropdown"] ul::-webkit-scrollbar-thumb,
    [data-testid="stSelectboxVirtualDropdown"] div::-webkit-scrollbar-thumb,
    div[data-baseweb="popover"] ul::-webkit-scrollbar-thumb,
    div[data-baseweb="popover"] div::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 4px;
    }

    /* ---------- BUTTONS ---------- */
    .stFormSubmitButton > button,
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        background: var(--btn-bg) !important;
        color: var(--btn-text) !important;
        font-weight: 700;
        font-size: clamp(1rem, 2.5vw, 1.15rem);
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 8px 24px rgba(37,99,235,0.35);
        transition: transform 0.15s ease, background-color 0.2s ease, box-shadow 0.2s ease;
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
        background: #1D4ED8 !important;
        color: var(--btn-text) !important;
        border: none;
        box-shadow: 0 10px 28px rgba(37,99,235,0.45);
    }
    .stFormSubmitButton > button p,
    div[data-testid="stButton"] > button p,
    div[data-testid="stDownloadButton"] > button p {
        color: var(--btn-text) !important;
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
        color: var(--heading);
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    .positive .result-title {color: var(--bad);}
    .negative .result-title {color: var(--good);}

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
        background: var(--card);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        aspect-ratio: 1 / 1;
    }
    .gauge-val {
        font-size: clamp(1.4rem, 4.5vw, 2rem);
        font-weight: 800;
        color: var(--heading);
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
        color: var(--heading);
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
        color: var(--heading);
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
        f"<div class='side-title'>{tr('About')}</div>"
        f"<div class='side-text'>{tr('ABOUT_TEXT')}</div>"
        f"<div class='side-title'>{tr('How to use')}</div>"
        f"<div class='side-text'>{tr('HOWTO_TEXT')}</div>"
        f"<div class='side-title'>{tr('Disclaimer')}</div>"
        f"<div class='side-text'>{tr('DISCLAIMER_TEXT')}</div>",
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    "<div class='hero'>"
    f"<div class='hero-title'>❤️ {tr('Heart Disease Prediction System')}</div>"
    f"<div class='hero-sub'>{tr('Enter the 13 patient details to estimate the risk of heart disease')}</div>"
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

FIELDS = [
    ("age", "Age"), ("gender", "Gender"), ("cp", "Chest Pain Type"),
    ("trestbps", "Resting Blood Pressure"), ("chol", "Cholesterol"),
    ("fbs", "Fasting Blood Sugar > 120 mg/dl"), ("restecg", "Resting ECG Result"),
    ("thalach", "Maximum Heart Rate Achieved"), ("exang", "Exercise Induced Angina"),
    ("oldpeak", "Old Peak (ST Depression)"), ("slope", "Slope of Peak Exercise ST"),
    ("ca", "Number of Major Vessels"), ("thal", "Thalassemia"),
]

DEFAULTS = {
    "age": 50, "gender": "Male", "cp": "Typical Angina", "trestbps": 120,
    "chol": 200, "fbs": "No", "restecg": "Normal", "thalach": 150,
    "exang": "No", "oldpeak": 1.0, "slope": "Upsloping", "ca": "0",
    "thal": "Normal",
}


def clear_form():
    """Reset every input to default values."""
    for code in LANGUAGES.values():
        for key, value in DEFAULTS.items():
            st.session_state[f"{key}_{code}"] = value


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
    result_style = ParagraphStyle(
        "r", parent=styles["Heading2"], textColor=colour, alignment=1)
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
        Paragraph("Patient details used for this prediction",
                  styles["Heading3"]),
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

    def wk(name):
        return f"{name}_{lang}"

    previous = st.session_state.get("form_lang")
    for key, value in DEFAULTS.items():
        if previous and previous != lang and f"{key}_{previous}" in st.session_state:
            st.session_state[wk(key)] = st.session_state[f"{key}_{previous}"]
        else:
            st.session_state.setdefault(wk(key), value)
    st.session_state.form_lang = lang

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
            s = {key: st.session_state[wk(key)] for key in DEFAULTS}
            sex = 1 if s["gender"] == "Male" else 0

            patient = np.array([[
                float(s["age"]),
                sex,
                CP_MAP[s["cp"]],
                float(s["trestbps"]),
                float(s["chol"]),
                FBS_MAP[s["fbs"]],
                RESTECG_MAP[s["restecg"]],
                float(s["thalach"]),
                EXANG_MAP[s["exang"]],
                float(s["oldpeak"]),
                SLOPE_MAP[s["slope"]],
                int(s["ca"]),
                THAL_MAP[s["thal"]]
            ]])

            patient_scaled = scaler.transform(patient)

            with st.spinner("Analysing patient data..."):
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
                title, css_class, conf = f"🔴 {tr('Heart Disease Detected')}", "positive", probability * 100
            else:
                title, css_class, conf = f"🟢 {tr('No Heart Disease Detected')}", "negative", (
                    1 - probability) * 100

            st.session_state.result = {
                "title": title, "css_class": css_class, "risk_pct": risk_pct,
                "conf": conf, "level": level,
                "inputs": {
                    "Age": s["age"], "Gender": s["gender"], "Chest Pain Type": s["cp"],
                    "Resting Blood Pressure": s["trestbps"], "Cholesterol": s["chol"],
                    "Fasting Blood Sugar > 120": s["fbs"], "Resting ECG": s["restecg"],
                    "Max Heart Rate": s["thalach"], "Exercise Angina": s["exang"],
                    "Old Peak": f"{s['oldpeak']:.2f}", "Slope": s["slope"],
                    "Major Vessels": s["ca"], "Thalassemia": s["thal"],
                },
            }
            st.session_state.page = "result"
            st.rerun()

        except Exception as e:
            st.error(f"{tr('Something went wrong')}: {e}")


# ============================================================
# PAGE 2: RESULT
# ============================================================

else:
    r = st.session_state.result
    if r is None:
        # session was reset (e.g. server restart): go back to the form
        st.session_state.page = "form"
        st.rerun()
    color = t[LEVEL_COLOUR[r["level"]]]

    _, centre, _ = st.columns([1, 2, 1])

    with centre:
        st.markdown(
            f"<div class='result-card {r['css_class']}'>"
            f"<div class='result-title'>{r['title']}</div>"
            f"<div class='gauge' style='--p:{r['risk_pct']:.1f}; --c:{color};'>"
            f"<div class='gauge-inner'>"
            f"<div class='gauge-val'>{r['risk_pct']:.1f}%</div>"
            f"<div class='gauge-lbl'>{tr('Disease probability')}</div>"
            f"</div></div>"
            f"<div class='metrics'>"
            f"<div class='metric'><div class='m-val'>{r['conf']:.1f}%</div><div class='m-lbl'>{tr('Confidence')}</div></div>"
            f"<div class='metric'><div class='m-val' style='color:{color};'>{tr(r['level'])}</div><div class='m-lbl'>{tr('Risk level')}</div></div>"
            f"</div>"
            f"<div class='note'>{tr('This tool supports screening only and is not a medical diagnosis.')}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    rows = "".join(
        f"<div class='sum-row'><span>{tr(k)}</span><b>{tr(str(v))}</b></div>"
        for k, v in r["inputs"].items()
    )
    st.markdown(
        f"<div class='sum-card'><div class='sum-title'>📋 {tr('Patient details used for this prediction')}</div>"
        f"<div class='sum-grid'>{rows}</div></div>",
        unsafe_allow_html=True
    )

    # values outside the typical healthy range (display only, not used by the model)
    inp = r["inputs"]
    flags = []
    if inp["Resting Blood Pressure"] >= 130:
        flags.append(f"Resting blood pressure is high ({inp['Resting Blood Pressure']} mm Hg)")
    if inp["Cholesterol"] >= 240:
        flags.append(f"Cholesterol is high ({inp['Cholesterol']} mg/dl)")
    if inp["Fasting Blood Sugar > 120"] == "Yes":
        flags.append("Fasting blood sugar is above 120 mg/dl")
    if inp["Exercise Angina"] == "Yes":
        flags.append("Chest pain occurs during exercise")
    if float(inp["Old Peak"]) >= 2:
        flags.append(f"ST depression is elevated ({inp['Old Peak']})")
    if int(inp["Major Vessels"]) > 0:
        flags.append(f"{inp['Major Vessels']} major vessel(s) affected")

    _, flag_col, _ = st.columns([1, 2, 1])
    with flag_col:
        if flags:
            st.warning("**Values outside the typical range:**\n\n" +
                       "\n".join(f"- {f}" for f in flags))
        else:
            st.success("All key measurements are within the typical range.")

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        st.download_button(
            f"📄 {tr('Download report (PDF)')}",
            data=build_report_pdf(r),
            file_name=f"heart_report_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            width="stretch",
        )
        if st.button("← New prediction", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()
