from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    joblib = None
    JOBLIB_AVAILABLE = False
from PIL import Image

st.set_page_config(page_title="AI Konsentrasi & Engagement", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'



# ==========
# Konfigurasi skor (ubah jika mau)
# ==========
# urutan kelas: Low, Medium, High
SCORE_MAP = np.array([30, 60, 90])

def prob_to_score(prob_vec):
    return float(np.dot(prob_vec, SCORE_MAP))

def score_to_label_id(score):
    if score < 50: return "Low"
    if score < 75: return "Medium"
    return "High"

def id_to_indo(lbl):
    return {"Low":"rendah","Medium":"sedang","High":"tinggi"}[lbl]

def rekomendasi(conc_score, eng_score):
    recs = []
    # konsentrasi
    if conc_score < 50:
        recs += ["kuis singkat 2-3 menit", "polling/cek pemahaman cepat"]
    elif conc_score < 75:
        recs += ["contoh kasus + pertanyaan pemantik"]
    else:
        recs += ["lanjutkan materi (fokus baik)"]
    # engagement
    if eng_score < 50:
        recs += ["diskusi singkat/breakout room"]
    elif eng_score < 75:
        recs += ["minta respon chat/emoji"]
    else:
        recs += ["pertahankan format pembelajaran"]
    # unik
    recs = list(dict.fromkeys(recs))
    return "; ".join(recs)

@st.cache_resource
def load_engagement_model():
    path = MODELS_DIR / 'engagement_model.joblib'
    if not path.exists():
        st.warning(f'Model engagement tidak ditemukan: {path}. Menggunakan fallback.')
        return None
    if not JOBLIB_AVAILABLE:
        st.info("Modul `joblib` tidak terpasang di environment. Menggunakan prediksi fallback.")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Gagal memuat model engagement: {e}. Menggunakan fallback.")
        return None

@st.cache_resource
def get_tensorflow():
    try:
        import tensorflow as tf
        return tf
    except Exception:
        return None

@st.cache_resource
def load_concentration_model():
    tf = get_tensorflow()
    if tf is None:
        return None
    path = MODELS_DIR / 'concentration_model.keras'
    if not path.exists():
        st.error(f'Model konsentrasi tidak ditemukan: {path}')
        return None
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model konsentrasi: {e}")
        return None

ENG_MODEL = load_engagement_model()
IMG_MODEL = None
IMG_SIZE = (160, 160)
CLASS_ORDER = ["Low","Medium","High"]  # pastikan konsisten

st.title("Aplikasi Prediksi Konsentrasi & Engagement Mahasiswa (SDG 4)")

tab1, tab2, tab3 = st.tabs(["Prediksi Engagement (Tabular)", "Prediksi Konsentrasi (Gambar)", "Gabungan + Rekomendasi"])

def clamp(value, min_value=0.0, max_value=100.0):
    return max(min_value, min(max_value, value))

def fallback_engagement_score(row):
    raw = (
        row["time_spent_weekly"] * 4
        + row["quiz_score_avg"] * 0.4
        + row["video_watched_percent"] * 0.3
        + row["assignments_submitted"] * 8
        + row["login_frequency"] * 3
        + row["session_duration_avg"] * 0.6
        + row["forum_posts"] * 4
    )
    return clamp(raw)

def fallback_concentration_prob(image_rgb):
    gray = np.mean(image_rgb, axis=2)
    mean = float(np.mean(gray))
    if mean < 85:
        return np.array([0.7, 0.2, 0.1])
    if mean < 170:
        return np.array([0.2, 0.6, 0.2])
    return np.array([0.1, 0.2, 0.7])

# =========================
# TAB 1: Engagement Tabular
# =========================
with tab1:
    st.subheader("Input Perilaku Mahasiswa (Dataset #1)")
    c1, c2, c3 = st.columns(3)

    with c1:
        time_spent_weekly = st.number_input("time_spent_weekly", min_value=0.0, value=8.0, step=0.5)
        quiz_score_avg = st.number_input("quiz_score_avg", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        forum_posts = st.number_input("forum_posts", min_value=0, value=2, step=1)

    with c2:
        video_watched_percent = st.number_input("video_watched_percent", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
        assignments_submitted = st.number_input("assignments_submitted", min_value=0, value=3, step=1)
        login_frequency = st.number_input("login_frequency", min_value=0, value=5, step=1)

    with c3:
        session_duration_avg = st.number_input("session_duration_avg", min_value=0.0, value=25.0, step=1.0)
        device_type = st.selectbox("device_type", ["Desktop","Mobile","Tablet"])
        course_difficulty = st.selectbox("course_difficulty", ["Easy","Medium","Hard"])
        region = st.selectbox("region", ["Urban","Suburban","Rural"])

    X = pd.DataFrame([{
        "time_spent_weekly": time_spent_weekly,
        "quiz_score_avg": quiz_score_avg,
        "forum_posts": forum_posts,
        "video_watched_percent": video_watched_percent,
        "assignments_submitted": assignments_submitted,
        "login_frequency": login_frequency,
        "session_duration_avg": session_duration_avg,
        "device_type": device_type,
        "course_difficulty": course_difficulty,
        "region": region
    }])

    if ENG_MODEL is None:
        st.info("Mode fallback aktif: model engagement tidak tersedia di environment ini.")
    if st.button("Prediksi Engagement"):
        if ENG_MODEL is None:
            eng_score = fallback_engagement_score(X.iloc[0])
            eng_cat = score_to_label_id(eng_score)
            st.success(f"Engagement: {id_to_indo(eng_cat)} ({eng_score:.0f})")
            st.write("Probabilitas (Low/Medium/High):", "fallback")
        elif not hasattr(ENG_MODEL, "predict_proba"):
            st.error("Model engagement tidak mendukung prediksi probabilitas (predict_proba).")
        else:
            probs = ENG_MODEL.predict_proba(X)[0]
            # pastikan urutan kelas sesuai Low/Medium/High
            model_classes = list(getattr(ENG_MODEL, "classes_", CLASS_ORDER))
            try:
                probs_ordered = np.array([probs[model_classes.index(c)] for c in CLASS_ORDER])
            except ValueError:
                st.error("Urutan kelas pada model engagement tidak sesuai (Low/Medium/High).")
                st.stop()

            eng_score = prob_to_score(probs_ordered)
            eng_cat = score_to_label_id(eng_score)

            st.success(f"Engagement: {id_to_indo(eng_cat)} ({eng_score:.0f})")
            st.write("Probabilitas (Low/Medium/High):", probs_ordered.round(3))

# =========================
# TAB 2: Konsentrasi Gambar
# =========================
with tab2:
    st.subheader("Upload Gambar Wajah Mahasiswa (Dataset #2)")
    tf = get_tensorflow()
    if tf is None:
        st.info("TensorFlow tidak tersedia. Menggunakan prediksi fallback berbasis intensitas gambar.")
    up = st.file_uploader("Upload JPG/PNG", type=["jpg","jpeg","png"])

    if up is not None:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Input image", use_column_width=True)

        x = np.array(img.resize(IMG_SIZE)) / 255.0
        x = np.expand_dims(x, axis=0)

        model = load_concentration_model()

        if model is None:
            prob = fallback_concentration_prob(np.array(img))
            st.warning("Model konsentrasi tidak tersedia. Menggunakan fallback sederhana.")
        else:
            prob = model.predict(x, verbose=0)[0]  # Low/Medium/High

        conc_score = prob_to_score(prob)
        conc_cat = score_to_label_id(conc_score)

        st.success(f"Konsentrasi: {id_to_indo(conc_cat)} ({conc_score:.0f})")
        st.write("Probabilitas (Low/Medium/High):", prob.round(3))

# =========================
# TAB 3: Gabungan + Rekomendasi
# =========================
with tab3:
    st.subheader("Gabungan Prediksi + Rekomendasi Adaptif")
    st.info("Isi Engagement (tab 1) dan upload gambar (tab 2). Lalu masukkan skor manual jika perlu.")

    colA, colB = st.columns(2)
    with colA:
        conc_score_in = st.number_input("Skor Konsentrasi (0-100)", min_value=0, max_value=100, value=30, step=1)
    with colB:
        eng_score_in = st.number_input("Skor Engagement (0-100)", min_value=0, max_value=100, value=31, step=1)

    if st.button("Buat Rekomendasi"):
        rec = rekomendasi(conc_score_in, eng_score_in)
        st.write(f"Output: **Konsentrasi = {id_to_indo(score_to_label_id(conc_score_in))} ({conc_score_in})**, "
                 f"**Engagement = {id_to_indo(score_to_label_id(eng_score_in))} ({eng_score_in})**")
        st.warning(f"Rekomendasi: {rec}")
