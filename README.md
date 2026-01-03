# AI Konsentrasi & Engagement Mahasiswa (SDG 4)

Streamlit app untuk memprediksi:
## Cara menjalankan (lokal)

1. Pastikan Python 3.8+ terpasang. Di Windows gunakan `py -3` atau perintah `python` jika tersedia di PATH.
2. Masuk ke folder proyek:

```powershell
cd "C:\Users\HP\Documents\GitHub\ai-konsentrasi-engagement"
```

3. Pasang dependencies:

```powershell
py -3 -m pip install -r requirements.txt
```

4. Jalankan Streamlit:

```powershell
py -3 -m streamlit run app.py
```

Catatan: `requirements.txt` sekarang mencantumkan `joblib`. Jika Anda menggunakan virtual environment, aktifkan terlebih dahulu.

## Catatan deploy (Streamlit Cloud)

- Setelah push ke repository, Streamlit Cloud akan mencoba menginstal paket dari `requirements.txt` dan menjalankan `app.py`.
- Jika deployment masih melaporkan `ModuleNotFoundError` untuk paket tertentu, periksa log deploy (Manage app → Logs) untuk paket yang gagal terpasang dan versi Python yang digunakan oleh Cloud.

Jika Anda mau, saya bisa menambahkan instruksi pembuatan virtualenv atau petunjuk khusus Windows.
