
import streamlit as st
import pickle
import re
import nltk
import spacy
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

st.set_page_config(page_title="Classifier", page_icon="🔍", layout="centered")

# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model_stage1.pkl", "rb") as f:
        model_stage1 = pickle.load(f)
    with open("model_stage2.pkl", "rb") as f:
        model_stage2 = pickle.load(f)
    with open("encoder_stage2.pkl", "rb") as f:
        encoder_stage2 = pickle.load(f)
    return vectorizer, model_stage1, model_stage2, encoder_stage2

vectorizer, model_stage1, model_stage2, encoder_stage2 = load_models()

# ── Setup preprocessing ────────────────────────────────────
stemmer_id = StemmerFactory().create_stemmer()
nlp_en = spacy.load("en_core_web_sm")
stop_wordsEN = set(stopwords.words("english"))
stop_wordsID = set(stopwords.words("indonesian"))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def detect_language_safe(text):
    s = str(text).strip()
    if len(re.sub(r'[^A-Za-z]+', '', s)) < 3:
        return "en"
    try:
        lang = detect(s)
        return lang if lang in ["en", "id"] else "en"
    except LangDetectException:
        return "en"


slang_dict = {
    # Kata ganti & partikel
    "yg": "yang", "dg": "dengan", "tdk": "tidak", "gak": "tidak",
    "gk": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak",
    "tak": "tidak", "tp": "tapi", "tpi": "tapi", "krn": "karena",
    "krna": "karena", "karna": "karena", "krena": "karena",
    "utk": "untuk", "u": "untuk", "buat": "untuk", "spy": "supaya",
    "biar": "supaya", "sm": "sama", "ama": "sama", "udh": "sudah",
    "udah": "sudah", "dah": "sudah", "sdh": "sudah", "blm": "belum",
    "blum": "belum", "msh": "masih", "lg": "lagi", "lgi": "lagi",
    "jg": "juga", "jga": "juga", "sj": "saja", "aja": "saja",
    "aj": "saja", "jd": "jadi", "jdi": "jadi", "sdg": "sedang",
    "lgi": "sedang", "klo": "kalau", "klu": "kalau", "kl": "kalau",
    "kalo": "kalau", "kpd": "kepada", "pd": "pada", "dr": "dari",
    "dri": "dari", "dgn": "dengan", "ttg": "tentang",

    # Sapaan & kata umum
    "gw": "saya", "gue": "saya", "w": "saya", "aq": "saya",
    "ak": "saya", "aku": "saya", "lo": "kamu", "lu": "kamu",
    "elo": "kamu", "loe": "kamu", "km": "kamu", "kmu": "kamu",
    "dy": "dia", "doi": "dia", "mrk": "mereka", "kt": "kita",
    "qt": "kita", "kmi": "kami",

    # Ekspresi & slang umum
    "bgt": "banget", "bngt": "banget", "bget": "banget",
    "bgt": "banget", "skrg": "sekarang", "skrng": "sekarang",
    "skg": "sekarang", "trs": "terus", "trus": "terus",
    "jgn": "jangan", "jngn": "jangan", "mau": "mau",
    "mw": "mau", "mo": "mau", "bs": "bisa", "bsa": "bisa",
    "hrs": "harus", "hrs": "harus", "pls": "tolong",
    "tlg": "tolong", "mksh": "terima kasih", "makasih": "terima kasih",
    "thx": "terima kasih", "tks": "terima kasih",

    # Kata sifat & ekspresi
    "byk": "banyak", "bnyk": "banyak", "dikit": "sedikit",
    "sdikit": "sedikit", "bsr": "besar", "kecil": "kecil",
    "gede": "besar", "gde": "besar", "lbh": "lebih",
    "lebih": "lebih", "plg": "paling", "paling": "paling",
    "emg": "memang", "emng": "memang", "mmg": "memang",
    "memang": "memang", "knp": "kenapa", "ngp": "kenapa",
    "knapa": "kenapa", "gmn": "bagaimana", "gmna": "bagaimana",
    "gimana": "bagaimana", "bgmn": "bagaimana",

    # Kata kerja umum
    "bilang": "berkata", "blg": "berkata", "makan": "makan",
    "mkn": "makan", "pergi": "pergi", "prgi": "pergi",
    "dtg": "datang", "dateng": "datang",

    # Kata negatif / bullying related
    "bodo": "bodoh", "blo'on": "bodoh", "bloon": "bodoh",
    "blo on": "bodoh", "tolol": "bodoh", "bangsat": "bajingan",
    "bgs": "bagus", "jelek": "jelek", "jlek": "jelek",
    "ancur": "hancur", "hancurin": "menghancurkan",
}

def normalize_slang(text):
    if not isinstance(text, str):
        return text
    tokens = text.split()
    normalized = [slang_dict.get(word.lower(), word) for word in tokens]
    return ' '.join(normalized)

def preprocess_text(text):
    lang = detect_language_safe(text)
    cleaned = clean_text(text)

    if lang == "id":
        cleaned = normalize_slang(cleaned)  # normalisasi slang ID

    text_fixed = contractions.fix(cleaned)
    tokens = word_tokenize(text_fixed)

    if lang == "id":
        tokens = [stemmer_id.stem(w) for w in tokens if w not in stop_wordsID]
    else:
        doc = nlp_en(cleaned)
        tokens = [token.lemma_ for token in doc if token.text not in stop_wordsEN]

    return ' '.join(tokens)

def predict_hierarchical(text):
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])
    pred_s1 = model_stage1.predict(vec)[0]
    if pred_s1 == 0:
        return "not_cyberbullying", None
    pred_s2 = model_stage2.predict(vec)[0]
    label = encoder_stage2.inverse_transform([pred_s2])[0]
    return "cyberbullying", label

# ── UI ─────────────────────────────────────────────────────
st.title("🔍 Cyberbullying Tweet Classifier")
st.markdown("**Final Project — Group 3 (Baby Python) | Data Science Batch 59 Digital Skola**")
st.markdown("---")

st.markdown("### Masukkan teks tweet di bawah ini:")
user_input = st.text_area("", placeholder="Ketik tweet di sini...", height=150)

color_map = {
    "age":                "🟠",
    "ethnicity":          "🟣",
    "gender":             "🔵",
    "religion":           "🟡",
    "other_cyberbullying":"🔴"
}

if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong!")
    else:
        with st.spinner("Menganalisis tweet..."):
            result, category = predict_hierarchical(user_input)

        st.markdown("---")
        st.markdown("### Hasil Prediksi:")

        if result == "not_cyberbullying":
            st.success("🟢 **Not Cyberbullying**")
            st.info("✅ Tweet ini tidak mengandung unsur cyberbullying.")
        else:
            emoji = color_map.get(category, "⚪")
            st.error("⚠️ Tweet ini terdeteksi mengandung cyberbullying!")
            st.markdown(f"**Kategori:** {emoji} **{category.replace('_', ' ').title()}**")
