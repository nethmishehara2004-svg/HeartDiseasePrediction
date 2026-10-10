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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from tensorflow.keras.models import load_model

from backgrounds import background_css

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
# TRANSLATIONS (English text is the key; English needs no entry)
# Option values sent to the model always stay English, only the
# text shown on screen is translated.
# ============================================================

LANGUAGES = {"සිංහල": "si", "தமிழ்": "ta", "English": "en"}   # order = priority; first is the default

TRANSLATIONS = {
    "si": {
        # app chrome
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
        # form
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
        "Fasting Blood Sugar > 120 mg/dL": "නිරාහාර රුධිර සීනි > 120 mg/dL",
        "Yes": "ඔව්",
        "No": "නැත",
        "Resting ECG Result": "විවේක ECG ප්‍රතිඵලය",
        "Normal": "සාමාන්‍ය",
        "ST-T Abnormality": "ST-T අසාමාන්‍යතාව",
        "Left Ventricular Hypertrophy": "වම් හෘද කෝෂිකාවේ අධිවර්ධනය",
        "Maximum Heart Rate Achieved": "ළඟා වූ උපරිම හෘද ස්පන්දන වේගය",
        "Exercise Induced Angina": "ව්‍යායාමයෙන් ඇති වන ඇන්ජයිනාව",
        "Old Peak (ST Depression)": "Old Peak (ST අවපීඩනය)",
        "Slope of Peak Exercise ST": "උපරිම ව්‍යායාමයේදී ST කොටසේ බෑවුම",
        "Upsloping": "ඉහළට බෑවුම්",
        "Flat": "සමතලා",
        "Downsloping": "පහළට බෑවුම්",
        "Number of Major Vessels": "ප්‍රධාන රුධිර නාල ගණන",
        "Thalassemia": "තැලසීමියාව",
        "Fixed Defect": "ස්ථීර දෝෂය",
        "Reversible Defect": "ප්‍රතිවර්ත්‍ය දෝෂය",
        # result
        "Low risk of heart disease": "හෘද රෝග අවදානම අඩුයි",
        "Moderate risk of heart disease": "හෘද රෝග අවදානම මධ්‍යමයි",
        "High risk of heart disease": "හෘද රෝග අවදානම ඉහළයි",
        "Disease probability": "රෝග සම්භාවිතාව",
        "Model certainty": "ආකෘතියේ නිශ්චිතභාවය",
        "Risk level": "අවදානම් මට්ටම",
        "Low": "අඩු",
        "Moderate": "මධ්‍යම",
        "High": "ඉහළ",
        "Patient details used for this prediction": "මෙම පුරෝකථනයට භාවිතා කළ රෝගී තොරතුරු",
        # PDF
        "Something went wrong": "යම් දෝෂයක් සිදු විය",
        "PDF font note": "මෙම භාෂාව සඳහා PDF අකුරු 'fonts' ෆෝල්ඩරයේ නොමැත; වාර්තාව ඉංග්‍රීසියෙන් සකස් වේ.",
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
        "Fasting Blood Sugar > 120 mg/dL": "உண்ணாவிரத இரத்தச் சர்க்கரை > 120 mg/dL",
        "Yes": "ஆம்",
        "No": "இல்லை",
        "Resting ECG Result": "ஓய்வு நிலை ECG முடிவு",
        "Normal": "இயல்பானது",
        "ST-T Abnormality": "ST-T அசாதாரணம்",
        "Left Ventricular Hypertrophy": "இடது இதயக் கீழறை தசை தடிப்பு",
        "Maximum Heart Rate Achieved": "அடைந்த அதிகபட்ச இதயத் துடிப்பு",
        "Exercise Induced Angina": "உடற்பயிற்சியால் ஏற்படும் ஆஞ்சினா",
        "Old Peak (ST Depression)": "Old Peak (ST தாழ்வு)",
        "Slope of Peak Exercise ST": "உச்ச உடற்பயிற்சியின் போது ST பகுதியின் சரிவு",
        "Upsloping": "மேல்நோக்கிய சரிவு",
        "Flat": "தட்டையான சரிவு",
        "Downsloping": "கீழ்நோக்கிய சரிவு",
        "Number of Major Vessels": "முக்கிய இரத்த நாளங்களின் எண்ணிக்கை",
        "Thalassemia": "தலசீமியா",
        "Fixed Defect": "நிலையான குறைபாடு",
        "Reversible Defect": "மீளக்கூடிய குறைபாடு",
        "Low risk of heart disease": "இதய நோய் ஆபத்து குறைவு",
        "Moderate risk of heart disease": "இதய நோய் ஆபத்து மிதமானது",
        "High risk of heart disease": "இதய நோய் ஆபத்து அதிகம்",
        "Disease probability": "நோய் நிகழ்தகவு",
        "Model certainty": "மாதிரியின் உறுதிநிலை",
        "Risk level": "ஆபத்து நிலை",
        "Low": "குறைவு",
        "Moderate": "மிதமான",
        "High": "அதிகம்",
        "Patient details used for this prediction": "இந்தக் கணிப்புக்குப் பயன்படுத்தப்பட்ட நோயாளி விவரங்கள்",
        "Something went wrong": "ஏதோ தவறு நடந்துவிட்டது",
        "PDF font note": "இந்த மொழிக்கான PDF எழுத்துரு 'fonts' கோப்புறையில் இல்லை; அறிக்கை ஆங்கிலத்தில் இருக்கும்.",
        "Heart Disease Prediction Report": "இதய நோய் முன்கணிப்பு அறிக்கை",
        "Generated on": "உருவாக்கிய தேதி",
        "Please consult a qualified doctor for any medical decision.":
            "எந்த மருத்துவ முடிவுக்கும் தகுதிவாய்ந்த மருத்துவரை அணுகவும்.",
    },
}

# Default English wording for the longer sidebar blocks
EN_TEXT = {
    "PDF font note": "PDF font for this language not found in the 'fonts' folder; the report will be in English.",
    "ABOUT_TEXT": "An AI-based tool that estimates the likelihood of heart disease "
                  "from 13 clinical measurements using a trained neural network.",
    "HOWTO_TEXT": "1. Fill in both cards.<br>2. Click Predict.<br>3. The result opens on a new page.",
    "DISCLAIMER_TEXT": "This tool supports screening only. It is not a medical diagnosis. "
                       "Always consult a qualified doctor.",
}


def tr_to(text, lang):
    """Translate `text` into `lang`; fall back to English / the text itself."""
    if lang == "en":
        return EN_TEXT.get(text, text)
    return TRANSLATIONS[lang].get(text, EN_TEXT.get(text, text))


# ============================================================
# PAGE STATE (form page -> result page)
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "form"
    st.session_state.result = None


# ============================================================
# APPEARANCE SETTINGS (language, theme, background photo)
# ============================================================

# Calm navy / slate palette. One muted blue accent; red is kept for "High risk" only.
# Set USE_PHOTO = True to show background.jpg (under a heavy veil) instead of the
# plain gradient with faint ECG line art.
USE_PHOTO = False

THEMES = {
    "Dark": {
        "card": "rgba(17,29,48,0.90)", "solid": "#111d30", "card2": "#172640",
        "border": "#25354f", "text": "#e8eef7", "muted": "#9aabc2",
        "accent": "#6aa8f0", "accent_text": "#8dbcf5", "accent_rgb": "106,168,240",
        "btn": "#2f6fb8", "btn_hover": "#3a7cc8", "track": "#22324b",
        "good": "#4cc38f", "warn": "#e0b04a", "bad": "#ef7b78",
        "side": "#0d192c", "scheme": "dark",
        "shadow": "0 1px 2px rgba(0,0,0,.35), 0 12px 28px -16px rgba(0,0,0,.6)",
    },
    "Light": {
        "card": "rgba(255,255,255,0.94)", "solid": "#ffffff", "card2": "#eef3fa",
        "border": "#c3d2e6", "text": "#15233a", "muted": "#587089",
        "accent": "#2563a8", "accent_text": "#2563a8", "accent_rgb": "37,99,168",
        "btn": "#2563a8", "btn_hover": "#1d5190", "track": "#e1e9f4",
        "good": "#1f8a5f", "warn": "#a9741a", "bad": "#c8433f",
        "side": "#eef4fb", "scheme": "light",
        "shadow": "0 1px 2px rgba(21,35,58,.06), 0 12px 28px -18px rgba(21,35,58,.25)",
    },
}


# Language and theme sit in a bar above the project title
top_lang, top_theme = st.columns([3, 2], gap="large")

with top_lang:
    lang_name = st.radio("Language / භාෂාව / மொழி", list(LANGUAGES.keys()),
                         horizontal=True, key="lang_name")
lang = LANGUAGES[lang_name]


def tr(text):
    """Translate into the language currently chosen at the top of the page."""
    return tr_to(text, lang)


with top_theme:
    theme_name = st.radio(tr("Theme"), ["Dark", "Light"], horizontal=True,
                          format_func=tr, key="theme_name")

t = THEMES[theme_name]


# ============================================================
# STYLING (theme variables here, everything else in style.css)
# ============================================================

st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: {t['scheme']};
        --card:{t['card']}; --solid:{t['solid']}; --card2:{t['card2']};
        --border:{t['border']}; --text:{t['text']}; --muted:{t['muted']};
        --accent:{t['accent']}; --accent-text:{t['accent_text']}; --accent-rgb:{t['accent_rgb']};
        --btn:{t['btn']}; --btn-hover:{t['btn_hover']}; --track:{t['track']};
        --good:{t['good']}; --bad:{t['bad']}; --warn:{t['warn']}; --shadow:{t['shadow']};
    }}
    {background_css(theme_name, USE_PHOTO)}
    section[data-testid="stSidebar"] {{background: {t['side']};}}
    </style>
    """,
    unsafe_allow_html=True
)

with open(os.path.join(BASE_DIR, "style.css"), encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


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
    f"<div class='hero-title'><svg class='hero-icon' viewBox='0 0 24 24' aria-hidden='true'>"
    "<path d='M12 20.5S3.5 15.6 3.5 9.5C3.5 6.8 5.5 5 7.8 5c1.7 0 3.2.9 4.2 2.4C13 5.9 14.5 5 16.2 5c2.3 0 4.3 1.8 4.3 4.5 0 6.1-8.5 11-8.5 11z' "
    "fill='none' stroke='currentColor' stroke-width='1.6' stroke-linejoin='round'/>"
    "<path d='M5 12h3l1.500-3 3 6 1.500-3h5' fill='none' stroke='currentColor' stroke-width='1.6' "
    "stroke-linecap='round' stroke-linejoin='round'/></svg>"
    f"{tr('Heart Disease Prediction System')}</div>"
    f"<div class='hero-sub'>{tr('Enter the 13 patient details to estimate the risk of heart disease')}</div>"
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# MAPPINGS (must match training encoding exactly)
# ============================================================

CP_MAP = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
FBS_MAP = {"No": 0, "Yes": 1}
RESTECG_MAP = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
EXANG_MAP = {"No": 0, "Yes": 1}
SLOPE_MAP = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
THAL_MAP = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

LEVEL_COLOUR = {"Low": "good", "Moderate": "warn", "High": "bad"}
LEVEL_CSS = {"Low": "negative", "Moderate": "moderate", "High": "positive"}
LEVEL_TITLE = {
    "Low": "Low risk of heart disease",
    "Moderate": "Moderate risk of heart disease",
    "High": "High risk of heart disease",
}

# (form key, English label) in the order shown on the result page / PDF
FIELDS = [
    ("age", "Age"), ("gender", "Gender"), ("cp", "Chest Pain Type"),
    ("trestbps", "Resting Blood Pressure"), ("chol", "Cholesterol"),
    ("fbs", "Fasting Blood Sugar > 120 mg/dL"), ("restecg", "Resting ECG Result"),
    ("thalach", "Maximum Heart Rate Achieved"), ("exang", "Exercise Induced Angina"),
    ("oldpeak", "Old Peak (ST Depression)"), ("slope", "Slope of Peak Exercise ST"),
    ("ca", "Number of Major Vessels"), ("thal", "Thalassemia"),
]

# Values the form returns to when "Clear form" is pressed
DEFAULTS = {
    "age": 50, "gender": "Male", "cp": "Typical Angina", "trestbps": 120,
    "chol": 200, "fbs": "No", "restecg": "Normal", "thalach": 150,
    "exang": "No", "oldpeak": 1.0, "slope": "Upsloping", "ca": "0",
    "thal": "Normal",
}


def clear_form():
    """Reset every input to its default value."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


# ============================================================
# DOWNLOADABLE REPORT (PDF)
# ============================================================

# Sinhala / Tamil text needs a font that contains those characters.
# Put these files in a "fonts" folder next to app.py:
#   NotoSansSinhala-Regular.ttf   NotoSansTamil-Regular.ttf
# (free from https://fonts.google.com/noto). If a file is missing, the PDF
# falls back to English.
PDF_FONT_FILES = {
    "si": ("NotoSansSinhala", "NotoSansSinhala-Regular.ttf"),
    "ta": ("NotoSansTamil", "NotoSansTamil-Regular.ttf"),
}


def pdf_fonts_for(language):
    """Return (regular, bold) font names usable for `language`, or None."""
    if language == "en":
        return "Helvetica", "Helvetica-Bold"
    font_name, file_name = PDF_FONT_FILES[language]
    path = os.path.join(BASE_DIR, "fonts", file_name)
    if not os.path.exists(path):
        return None
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font_name, path))
    return font_name, font_name


def build_report_pdf(r, language):
    """Return the prediction report as PDF bytes in `language`."""
    fonts = pdf_fonts_for(language)
    if fonts is None:
        language, fonts = "en", ("Helvetica", "Helvetica-Bold")
    regular, bold = fonts

    def T(text):
        return tr_to(text, language)

    colours = {"Low": "#1f8a5f", "Moderate": "#a9741a", "High": "#c8433f"}
    colour = colors.HexColor(colours[r["level"]])
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    styles = getSampleStyleSheet()
    grey = colors.HexColor("#587089")
    title_style = ParagraphStyle("t", parent=styles["Title"], fontName=bold,
                                 textColor=colors.HexColor("#2563a8"))
    meta_style = ParagraphStyle("m", parent=styles["Normal"], fontName=regular,
                                textColor=grey, alignment=1, spaceAfter=14)
    result_style = ParagraphStyle("r", parent=styles["Heading2"], fontName=bold,
                                  textColor=colour, alignment=1)
    head_style = ParagraphStyle("h", parent=styles["Heading3"], fontName=bold)
    note_style = ParagraphStyle("n", parent=styles["Normal"], fontName=regular,
                                fontSize=9, textColor=grey, leading=13)

    stats = Table(
        [[f"{r['risk_pct']:.1f}%", f"{r['conf']:.1f}%", T(r["level"])],
         [T("Disease probability"), T("Model certainty"), T("Risk level")]],
        colWidths=[5.5 * cm] * 3,
    )
    stats.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), bold),
        ("FONTNAME", (0, 1), (-1, 1), regular),
        ("FONTSIZE", (0, 0), (-1, 0), 18),
        ("TEXTCOLOR", (2, 0), (2, 0), colour),
        ("FONTSIZE", (0, 1), (-1, 1), 9),
        ("TEXTCOLOR", (0, 1), (-1, 1), grey),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3fa")),
        ("BOX", (0, 0), (-1, -1), 1.5, colour),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    rows = []
    for key, label in FIELDS:
        value = r["inputs"][key]
        shown = f"{value:.2f}" if key == "oldpeak" else str(value)
        rows.append([T(label), T(shown)])
    details = Table(rows, colWidths=[10 * cm, 6.5 * cm])
    details.setStyle(TableStyle([
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("FONTNAME", (0, 0), (0, -1), regular),
        ("FONTNAME", (1, 0), (1, -1), bold),
        ("TEXTCOLOR", (0, 0), (0, -1), grey),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#d8e1ee")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    story = [
        Paragraph(T("Heart Disease Prediction Report"), title_style),
        Paragraph(f"{T('Generated on')}: {generated}", meta_style),
        Paragraph(T(LEVEL_TITLE[r["level"]]), result_style),
        Spacer(1, 6),
        stats,
        Spacer(1, 18),
        Paragraph(T("Patient details used for this prediction"), head_style),
        details,
        Spacer(1, 24),
        Paragraph(T("This tool supports screening only and is not a medical diagnosis.") + " "
                  + T("Please consult a qualified doctor for any medical decision."), note_style),
    ]

    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, title=T("Heart Disease Prediction Report"),
                      leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm).build(story)
    return buffer.getvalue()


# ============================================================
# PAGE 1: INPUT FORM
# ============================================================

if st.session_state.page == "form":

    # make sure every input has a value (also after returning from the result page)
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    with st.form("patient_form"):

        col1, col2 = st.columns(2, gap="large")

        with col1:
            with st.container(border=True):
                st.markdown(f'<div class="field-group">{tr("Patient Information")}</div>',
                            unsafe_allow_html=True)
                st.number_input(tr("Age"), min_value=1, max_value=120, key="age")
                st.selectbox(tr("Gender"), ["Male", "Female"], format_func=tr, key="gender")
                st.selectbox(tr("Chest Pain Type"), list(CP_MAP), format_func=tr, key="cp")
                st.number_input(tr("Resting Blood Pressure"), min_value=80, max_value=250, key="trestbps")
                st.number_input(tr("Cholesterol"), min_value=100, max_value=600, key="chol")
                st.selectbox(tr("Fasting Blood Sugar > 120 mg/dL"), list(FBS_MAP),
                             format_func=tr, key="fbs")
                st.selectbox(tr("Resting ECG Result"), list(RESTECG_MAP),
                             format_func=tr, key="restecg")

        with col2:
            with st.container(border=True):
                st.markdown(f'<div class="field-group">{tr("Clinical Measurements")}</div>',
                            unsafe_allow_html=True)
                st.number_input(tr("Maximum Heart Rate Achieved"), min_value=60, max_value=220, key="thalach")
                st.selectbox(tr("Exercise Induced Angina"), list(EXANG_MAP),
                             format_func=tr, key="exang")
                st.number_input(tr("Old Peak (ST Depression)"), min_value=0.0, max_value=10.0,
                                step=0.1, key="oldpeak")
                st.selectbox(tr("Slope of Peak Exercise ST"), list(SLOPE_MAP),
                             format_func=tr, key="slope")
                st.selectbox(tr("Number of Major Vessels"), ["0", "1", "2", "3", "4"], key="ca")
                st.selectbox(tr("Thalassemia"), list(THAL_MAP), format_func=tr, key="thal")

        btn_predict, btn_clear = st.columns([3, 1])
        with btn_predict:
            submitted = st.form_submit_button(tr('Predict'), use_container_width=True)
        with btn_clear:
            st.form_submit_button(tr('Clear form'), on_click=clear_form,
                                  use_container_width=True)

    # ------------------------------------------------------------
    # PREDICTION (runs on submit, then moves to the result page)
    # ------------------------------------------------------------

    if submitted:
        try:
            s = st.session_state
            sex = 1 if s.gender == "Male" else 0

            patient = np.array([[
                float(s.age), sex, CP_MAP[s.cp], float(s.trestbps), float(s.chol),
                FBS_MAP[s.fbs], RESTECG_MAP[s.restecg], float(s.thalach),
                EXANG_MAP[s.exang], float(s.oldpeak), SLOPE_MAP[s.slope],
                int(s.ca), THAL_MAP[s.thal],
            ]])

            probability = float(model.predict(scaler.transform(patient), verbose=0)[0][0])
            risk_pct = probability * 100

            if risk_pct < 20:
                level = "Low"
            elif risk_pct < 50:
                level = "Moderate"
            else:
                level = "High"

            st.session_state.result = {
                "risk_pct": risk_pct,
                "conf": max(probability, 1 - probability) * 100,
                "level": level,
                # raw (English) values, translated only when displayed
                "inputs": {key: s[key] for key, _ in FIELDS},
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
    level = r["level"]
    color = t[LEVEL_COLOUR[level]]

    _, centre, _ = st.columns([1, 2, 1])

    with centre:
        st.markdown(
            f"<div class='result-card {LEVEL_CSS[level]}'>"
            f"<div class='result-title'><span class='dot'></span>{tr(LEVEL_TITLE[level])}</div>"
            f"<div class='gauge' style='--p:{r['risk_pct']:.1f}; --c:{color};'>"
            f"<div class='gauge-inner'>"
            f"<div class='gauge-val'>{r['risk_pct']:.1f}%</div>"
            f"<div class='gauge-lbl'>{tr('Disease probability')}</div>"
            f"</div></div>"
            f"<div class='metrics'>"
            f"<div class='metric'><div class='m-val'>{r['conf']:.1f}%</div>"
            f"<div class='m-lbl'>{tr('Model certainty')}</div></div>"
            f"<div class='metric'><div class='m-val' style='color:{color};'>{tr(level)}</div>"
            f"<div class='m-lbl'>{tr('Risk level')}</div></div>"
            f"</div>"
            f"<div class='note'>{tr('This tool supports screening only and is not a medical diagnosis.')}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    rows = ""
    for key, label in FIELDS:
        value = r["inputs"][key]
        shown = f"{value:.2f}" if key == "oldpeak" else str(value)
        rows += f"<div class='sum-row'><span>{tr(label)}</span><b>{tr(shown)}</b></div>"
    st.markdown(
        f"<div class='sum-card'><div class='sum-title'>{tr('Patient details used for this prediction')}</div>"
        f"<div class='sum-grid'>{rows}</div></div>",
        unsafe_allow_html=True
    )

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if lang != "en" and pdf_fonts_for(lang) is None:
            st.caption(tr("PDF font note"))
        st.download_button(
            tr('Download report (PDF)'),
            data=build_report_pdf(r, lang),
            file_name=f"heart_report_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        if st.button(tr('New prediction'), use_container_width=True):
            st.session_state.page = "form"
            st.rerun()
