import streamlit as st
import pandas as pd
import pickle

# ============================
#  LOAD MODEL
# ============================
with open("rf_model.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
preprocessor = data["preprocessor"]

st.set_page_config(page_title="Prediksi Diabetes", page_icon="🩺", layout="centered")

st.title("🩺 Dashboard Prediksi Diabetes")
st.write("Masukkan data pasien untuk memprediksi kemungkinan Diabetes Mellitus berdasarkan model Random Forest.")

# ============================
#  FORM INPUT PASIEN
# ============================
st.subheader("📋 Data Pasien")
usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=45)
tekanan_darah = st.number_input("Tekanan Darah (mmHg)", min_value=60, max_value=200, value=120)
insulin = st.number_input("Kadar Insulin", min_value=0.0, max_value=500.0, value=80.0)
glukosa_puasa = st.number_input("Glukosa Darah Puasa (mg/dL)", min_value=0.0, max_value=400.0, value=110.0)
hbA1c = st.number_input("Persentase HbA1c (%)", min_value=0.0, max_value=20.0, value=5.8)
kolesterol = st.number_input("Kolesterol (mg/dL)", min_value=50.0, max_value=400.0, value=180.0)
glukosa_sewaktu = st.number_input("Glukosa Darah Sewaktu (mg/dL)", min_value=0.0, max_value=400.0, value=140.0)
bbtb = st.number_input("BB/TB (BMI)", min_value=10.0, max_value=50.0, value=23.0)

pola_makan = st.selectbox("Pola Makan", ["SEHAT", "TIDAK SEHAT"])
jenis_kelamin = st.selectbox("Jenis Kelamin", ["LAKI-LAKI", "PEREMPUAN"])
bbtb_kat = st.selectbox("Kategori BB/TB", ["NORMAL", "OVERWEIGHT", "OBESITAS"])

# ============================
#  PREDIKSI
# ============================
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
        "POLA MAKAN": pola_makan,
        "JENIS KELAMIN": jenis_kelamin,
        "BB/TB": bbtb_kat
    }])

    X_input = preprocessor.transform(input_data)
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    st.subheader("🩸 Hasil Prediksi:")
    st.write(f"**Kemungkinan Diabetes:** {prob*100:.2f}%")
    if pred == 1:
        st.error("💀 Hasil: Positif Diabetes (DM)")
    else:
        st.success("💚 Hasil: Negatif (Tidak Terindikasi DM)")

    st.write("---")
    st.caption("Model menggunakan Random Forest dengan preprocessing otomatis (StandardScaler + OneHotEncoder).")
