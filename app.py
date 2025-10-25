import streamlit as st
import pandas as pd
import joblib
import os

# ============================
#  LOAD MODEL
# ============================
MODEL_PATH = "rf_model.pkl"

st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Dashboard Prediksi Diabetes")
st.markdown("""
Aplikasi ini memprediksi kemungkinan **Diabetes Mellitus (DM)** menggunakan model *Random Forest*  
berdasarkan hasil penelitian dan dataset klinis yang telah diproses.
""")

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ File model tidak ditemukan: `{MODEL_PATH}`. Pastikan file `rf_model.pkl` ada di folder yang sama dengan `app.py`.")
    st.stop()

try:
    data = joblib.load(MODEL_PATH)
    model = data["model"]
    preprocessor = data["preprocessor"]
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model `{MODEL_PATH}`: {e}")
    st.stop()


# ============================
#  FORM INPUT PASIEN
# ============================
st.header("📋 Masukkan Data Pasien")

col1, col2 = st.columns(2)

with col1:
    usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=45)
    tekanan_darah = st.number_input("Tekanan Darah (mmHg)", min_value=60, max_value=250, value=120)
    insulin = st.number_input("Kadar Insulin", min_value=0.0, max_value=500.0, value=85.0)
    glukosa_puasa = st.number_input("Glukosa Darah Puasa (mg/dL)", min_value=0.0, max_value=400.0, value=110.0)
    hbA1c = st.number_input("Persentase HbA1c (%)", min_value=0.0, max_value=20.0, value=6.0)

with col2:
    kolesterol = st.number_input("Kolesterol (mg/dL)", min_value=50.0, max_value=400.0, value=180.0)
    glukosa_sewaktu = st.number_input("Glukosa Darah Sewaktu (mg/dL)", min_value=0.0, max_value=400.0, value=150.0)
    bbtb = st.number_input("BB/TB (BMI)", min_value=10.0, max_value=50.0, value=23.0)

    # 🔹 Perubahan di sini: pilihan pola makan diganti ke frekuensi
    pola_makan = st.selectbox("Frekuensi Makan per Hari", [
        "1x/hari",
        "2x/hari",
        "3x/hari",
        "4x/hari",
        "5x/hari"
    ])

    jenis_kelamin = st.selectbox("Jenis Kelamin", ["LAKI-LAKI", "PEREMPUAN"])
    bbtb_kat = st.selectbox("Kategori BB/TB", ["NORMAL", "OVERWEIGHT", "OBESITAS"])


# ============================
#  PREDIKSI
# ============================
st.markdown("---")
if st.button("🔍 Prediksi Sekarang"):
    input_data = pd.DataFrame([{
        "USIA_num": usia,
        "Tekanan Darah (mmHg)_num": tekanan_darah,
        "INSULIN_num": insulin,
        "Glukosa Darah  Puasa (mg/dL)_num": glukosa_puasa,
        "Persentase kadar HbA1c (%)_num": hbA1c,
        "Kolesterol (mg/dL)_num": kolesterol,
        "Glukosa Darah  Sewaktu (mg/dL)_num": glukosa_sewaktu,
        "BB/TB_num": bbtb,
        "POLA MAKAN": pola_makan,         # tetap gunakan key yang sama
        "JENIS KELAMIN": jenis_kelamin,
        "BB/TB": bbtb_kat
    }])

    try:
        X_input = preprocessor.transform(input_data)
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]

        st.subheader("🩸 Hasil Prediksi:")
        st.metric(label="Kemungkinan Diabetes (%)", value=f"{prob*100:.2f}%")

        if pred == 1:
            st.error("💀 Hasil: **Positif Diabetes (DM)**")
        else:
            st.success("💚 Hasil: **Negatif (Tidak Terindikasi DM)**")

        st.markdown("---")
        st.caption("Model: Random Forest | Preprocessing: StandardScaler + OneHotEncoder")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat melakukan prediksi: {e}")
