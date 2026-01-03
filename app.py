import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
from pathlib import Path

# =========================
# Konfigurasi Path (aman untuk Streamlit Cloud)
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# =========================
# Konfigurasi App
# =========================
st.set_page_config(
    page_title="AI Konsentrasi & Engagement",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Prediksi Konsentrasi & Engagement")
st.caption("Demo Streamlit: prediksi engagement (tabular) dan konsentrasi (gambar) + rekomendasi pembelajaran adaptif.")

# =========================
# Load model (cache) — aman untuk Streamlit Cloud
# =========================
@st.cache_resource(show_spinner=False)
def load_engagement_model():
    path = MODELS_DIR / "engagement_model.joblib"
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_image_model():
    tf.get_logger().setLevel("ERROR")
    path = MODELS_DIR / "concentration_model.keras"
    return tf.keras.models.load_model(path)


@st.cache_resource(show_spinner=False)
def load_label_encoder():
    path = MODELS_DIR / "label_encoder.npy"
    return np.load(path, allow_pickle=True)


def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return arr


# Load once
try:
    ENGAGEMENT_MODEL = load_engagement_model()
    IMAGE_MODEL = load_image_model()
    LABEL_ENCODER = load_label_encoder()
except Exception as e:
    st.error(f"Gagal memuat model/dependensi: {e}")
    st.stop()

# =========================
# Input pilihan
# =========================
input_type = st.selectbox("Pilih tipe input:", ["Tabular (Perilaku/Interaksi)", "Image (Wajah/Visual)"])

# =========================
# TABULAR — Engagement
# =========================
if input_type.startswith("Tabular"):
    st.subheader("📊 Prediksi Engagement (Tabular)")

    # Sesuaikan field ini dengan dataset tabular Anda
    age = st.number_input("Age", min_value=10, max_value=80, value=21)
    gender = st.selectbox("Gender", ["Male", "Female"])
    lms_activity = st.number_input("LMS Activity (0–100)", min_value=0, max_value=100, value=60)
    discussion_posts = st.number_input("Discussion Posts", min_value=0, max_value=100, value=2)
    quiz_attempts = st.number_input("Quiz Attempts", min_value=0, max_value=50, value=1)

    # Contoh kategori lain
    internet_quality = st.selectbox("Internet Quality", ["Low", "Medium", "High"])
    device_type = st.selectbox("Device", ["Laptop", "Mobile", "Tablet"])

    if st.button("Prediksi Engagement"):
        # DataFrame harus sesuai skema training pipeline Anda.
        X_sample = pd.DataFrame(
            {
                "age": [age],
                "gender": [gender],
                "lms_activity": [lms_activity],
                "discussion_posts": [discussion_posts],
                "quiz_attempts": [quiz_attempts],
                "internet_quality": [internet_quality],
                "device_type": [device_type],
            }
        )

        try:
            eng_pred = ENGAGEMENT_MODEL.predict(X_sample)[0]
        except Exception as e:
            st.error(
                "Gagal prediksi. Pastikan nama kolom input di app.py sama dengan kolom yang dipakai saat training.\n"
                f"Detail error: {e}"
            )
            st.stop()

        # Konversi label ke skor 0–100 (contoh)
        score_map = {"Low": 30, "Medium": 60, "High": 85}
        engagement_score = int(score_map.get(str(eng_pred), 50))

        st.success(f"Engagement: **{eng_pred}** (skor: **{engagement_score}**) ")

        # Rekomendasi sederhana
        if engagement_score < 40:
            st.warning("Rekomendasi: kuis singkat 2–3 menit, polling cepat, atau breakout room singkat.")
        elif engagement_score < 70:
            st.info("Rekomendasi: variasikan aktivitas (diskusi + latihan singkat) dan berikan umpan balik cepat.")
        else:
            st.info("Rekomendasi: pertahankan ritme belajar, beri tantangan bertahap, dan materi lanjutan.")

# =========================
# IMAGE — Konsentrasi
# =========================
else:
    st.subheader("🖼️ Prediksi Konsentrasi (Image)")
    uploaded = st.file_uploader("Upload gambar wajah (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar input", use_column_width=True)

        if st.button("Prediksi Konsentrasi"):
            X_img = preprocess_image(img)
            y_prob = IMAGE_MODEL.predict(X_img)
            y_idx = int(np.argmax(y_prob, axis=1)[0])

            # LABEL_ENCODER berisi daftar kelas
            try:
                concentration_label = str(LABEL_ENCODER[y_idx])
            except Exception:
                concentration_label = str(y_idx)

            score_map = {"Low": 30, "Medium": 60, "High": 85}
            concentration_score = int(score_map.get(concentration_label, 50))

            st.success(f"Konsentrasi: **{concentration_label}** (skor: **{concentration_score}**) ")

            if concentration_score < 40:
                st.warning("Rekomendasi: kuis singkat 2–3 menit; cek pemahaman cepat; jeda singkat.")
            elif concentration_score < 70:
                st.info("Rekomendasi: beri ringkasan poin penting + latihan 3–5 soal.")
            else:
                st.info("Rekomendasi: lanjutkan materi lanjutan / studi kasus lebih menantang.")

st.divider()
st.caption("Catatan: ini demo. Untuk produksi, tambahkan logging, penyimpanan hasil ke DB, autentikasi, dan monitoring.")
