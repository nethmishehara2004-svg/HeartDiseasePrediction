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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
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
        "DISCLAIMER_TEXT": "මෙම මෙවලම පරීක්ෂණ සඳහා පමණක් උපකාරී වන අතර වෛද්‍ය රෝග විනිශ්චයක් නොවේ. සෑම විටම සුදුසුකම් ලත් වෛද්‍යවරයෙකුගෙන් උපදෙස් ලබාගන්න.",
        "This tool supports screening only and is not a medical diagnosis.":
            "මෙම මෙවලම පරීක්ෂණ සඳහා පමණක් උපකාරී වන අතර වෛද්‍ය රෝග විනිශ්චයක් නොවේ.",
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
        "Asymptomatic": "රෝග ලක්ෂණ නැත",
        "Resting Blood Pressure": "විවේක රුධිර පීඩනය",
        "Cholesterol": "කොලෙස්ටරෝල්",
        "Fasting Blood Sugar > 120 mg/dl": "නිරාහාර රුධිර සීනි > 120 mg/dl",
        "Yes": "ඔව්",
        "No": "නැත",
        "Resting ECG Result": "විවේක ECG ප්‍රතිඵලය",
        "Normal": "සාමාන්‍ය",
        "ST-T Abnormality": "ST-T අසාමාන්‍යතාව",
        "Left Ventricular Hypertrophy": "වම් නිලයේ අධිවර්ධනය",
        "Maximum Heart Rate Achieved": "ළඟා වූ උපරිම හෘද ස්පන්දන වේගය",
        "Exercise Induced Angina": "ව්‍යායාමයෙන් ඇති වන ඇන්ජයිනාව",
        "Old Peak (ST Depression)": "Old Peak (ST අවපීඩනය)",
        "Slope of Peak Exercise ST": "උපරිම ව්‍යායාම ST බෑවුම",
        "Upsloping": "ඉහළට බෑවුම්",
        "Flat": "සමතලා",
        "Downsloping": "පහළට බෑවුම්",
        "Number of Major Vessels": "ප්‍රධාන රුධිර නාල ගණන",
        "Thalassemia": "තැලසිමියාව",
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
        "DISCLAIMER_TEXT": "இந்தக் கருவி ஆரம்பநிலை பரிசோதனைக்கு மட்டுமே உதவுகிறது; இது மருத்துவ நோயறிதல் அல்ல. எப்போதும் தகுதிவாய்ந்த மருத்துவரை அணுகவும்.",
        "This tool supports screening only and is not a medical diagnosis.":
            "இந்தக் கருவி ஆரம்பநிலை பரிசோதனைக்கு மட்டுமே உதவுகிறது; இது மருத்துவ நோயறிதல் அல்ல.",
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
        "Asymptomatic": "அறிகுறியற்றது",
        "Resting Blood Pressure": "ஓய்வு நிலை இரத்த அழுத்தம்",
        "Cholesterol": "கொலஸ்ட்ரால்",
        "Fasting Blood Sugar > 120 mg/dl": "உண்ணாவிரத இரத்த சர்க்கரை > 120 mg/dl",
        "Yes": "ஆம்",
        "No": "இல்லை",
        "Resting ECG Result": "ஓய்வு நிலை ECG முடிவு",
        "Normal": "இயல்பானது",
        "ST-T Abnormality": "ST-T அசாதாரணம்",
        "Left Ventricular Hypertrophy": "இடது இதயக்கீழறை பெருக்கம்",
        "Maximum Heart Rate Achieved": "அடைந்த அதிகபட்ச இதயத் துடிப்பு",
        "Exercise Induced Angina": "உடற்பயிற்சியால் ஏற்படும் ஆஞ்சினா",
        "Old Peak (ST Depression)": "Old Peak (ST தாழ்வு)",
        "Slope of Peak Exercise ST": "உச்ச உடற்பயிற்சி ST சரிவு",
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
        "Heart Disease Prediction Report": "இதய நோய் முன்கணிப்பு அறிக்கை",
        "Generated on": "உருவாக்கிய தேதி",
        "Please consult a qualified doctor for any medical decision.":
            "எந்த மருத்துவ முடிவுக்கும் தகுதிவாய்ந்த மருத்துவரை அணுகவும்.",
    },
}

# Default English wording for the longer sidebar blocks
EN_TEXT = {
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


def load_background():
    """Return (mime, base64) for a background image next to app.py, if present."""
    for name, mime in (("background.jpg", "image/jpeg"),
                       ("background.jpeg", "image/jpeg"),
                       ("background.png", "image/png")):
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return mime, base64.b64encode(f.read()).decode()
    return None


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
bg = load_background()


# ============================================================
# STYLING (theme variables here, everything else in style.css)
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
    {bg_rule}
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
    f"<div class='hero-title'>❤️ {tr('Heart Disease Prediction System')}</div>"
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
LEVEL_EMOJI = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
LEVEL_TITLE = {
    "Low": "Low risk of heart disease",
    "Moderate": "Moderate risk of heart disease",
    "High": "High risk of heart disease",
}

# (form key, English label) in the order shown on the result page / PDF
FIELDS = [
    ("age", "Age"), ("gender", "Gender"), ("cp", "Chest Pain Type"),
    ("trestbps", "Resting Blood Pressure"), ("chol", "Cholesterol"),
    ("fbs", "Fasting Blood Sugar > 120 mg/dl"), ("restecg", "Resting ECG Result"),
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

    colours = {"Low": "#15935f", "Moderate": "#c79100", "High": "#d62839"}
    colour = colors.HexColor(colours[r["level"]])
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    styles = getSampleStyleSheet()
    grey = colors.HexColor("#5b6578")
    title_style = ParagraphStyle("t", parent=styles["Title"], fontName=bold,
                                 textColor=colors.HexColor("#e63946"))
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
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef1f7")),
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
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#d5dbe8")),
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
                st.markdown(f'<div class="field-group">🧍 {tr("Patient Information")}</div>',
                            unsafe_allow_html=True)
                st.number_input(tr("Age"), min_value=1, max_value=120, key="age")
                st.selectbox(tr("Gender"), ["Male", "Female"], format_func=tr, key="gender")
                st.selectbox(tr("Chest Pain Type"), list(CP_MAP), format_func=tr, key="cp")
                st.number_input(tr("Resting Blood Pressure"), min_value=80, max_value=250, key="trestbps")
                st.number_input(tr("Cholesterol"), min_value=100, max_value=600, key="chol")
                st.selectbox(tr("Fasting Blood Sugar > 120 mg/dl"), list(FBS_MAP),
                             format_func=tr, key="fbs")
                st.selectbox(tr("Resting ECG Result"), list(RESTECG_MAP),
                             format_func=tr, key="restecg")

        with col2:
            with st.container(border=True):
                st.markdown(f'<div class="field-group">🩺 {tr("Clinical Measurements")}</div>',
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
            submitted = st.form_submit_button(f"🔍 {tr('Predict')}", use_container_width=True)
        with btn_clear:
            st.form_submit_button(f"🧹 {tr('Clear form')}", on_click=clear_form,
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
            st.error(f"Something went wrong: {e}")


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
            f"<div class='result-title'>{LEVEL_EMOJI[level]} {tr(LEVEL_TITLE[level])}</div>"
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
        f"<div class='sum-card'><div class='sum-title'>📋 {tr('Patient details used for this prediction')}</div>"
        f"<div class='sum-grid'>{rows}</div></div>",
        unsafe_allow_html=True
    )

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if lang != "en" and pdf_fonts_for(lang) is None:
            st.caption(f"PDF font for this language not found in the 'fonts' folder; "
                       f"the report will be in English.")
        st.download_button(
            f"📄 {tr('Download report (PDF)')}",
            data=build_report_pdf(r, lang),
            file_name=f"heart_report_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        if st.button(f"← {tr('New prediction')}", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()
