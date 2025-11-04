# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------
# CONFIG
# -----------------------
# Ganti path ini bila perlu (nama file yang kamu simpan dari training)
#MODEL_CANDIDATES = [
#    "rf_model.pkl",
#    "rf_tuned_model_preproc.pkl",
#    "rf_model_preproc.pkl",
#]
MODEL_PATH = "rf_tuned_model_preproc.pkl"


st.set_page_config(page_title="Prediksi Diabetes", page_icon="🩺", layout="centered")
st.title("🩺 Dashboard Prediksi Diabetes")
st.markdown(
    """
Aplikasi ini memprediksi kemungkinan **Diabetes Mellitus (DM)** menggunakan model *Random Forest*.  
Pastikan file model (joblib) berada pada folder yang sama dengan `app.py`.
"""
)

# -----------------------
# LOAD MODEL (try several filenames)
# -----------------------
loaded = None
model_file = None
for fname in MODEL_PATH:
    if os.path.exists(fname):
        try:
            loaded = joblib.load(fname)
            model_file = fname
            break
        except Exception:
            # try next
            loaded = None

if loaded is None:
    st.error(
        "❌ File model tidak ditemukan atau gagal dimuat. \n\n"
        "Letakkan file model (joblib) di folder yang sama. Contoh nama file: `rf_model.pkl` atau `rf_tuned_model_preproc.pkl`."
    )
    st.stop()

# extract expected objects
model = loaded.get("model") or loaded.get("estimator") or None
preprocessor = loaded.get("preprocessor") or (loaded.get("model_pipeline").named_steps.get("preprocessor") if loaded.get("model_pipeline") else None)
feature_names = loaded.get("feature_names") or loaded.get("features") or loaded.get("original_feature_columns") or None
original_cols = loaded.get("original_feature_columns") or loaded.get("features") or None

if model is None or preprocessor is None:
    # try alternate structure: maybe the joblib is the pipeline itself
    if hasattr(loaded, "predict"):
        # user might have saved pipeline directly
        # assume pipeline: preprocessor = loaded.named_steps.get('preprocessor'), model from 'rf' or 'rf' named step
        try:
            preprocessor = loaded.named_steps.get("preprocessor")
            # try to get model step
            possible_model = None
            for n, step in loaded.named_steps.items():
                if n.lower() in ("rf","randomforestclassifier","model"):
                    possible_model = step
                    break
            model = possible_model or model
        except Exception:
            pass

if model is None or preprocessor is None:
    st.error("❌ Struktur file model tidak dikenali. Pastikan joblib berisi dict dengan keys: 'model' & 'preprocessor' atau sebuah pipeline.")
    st.stop()

st.success(f"✅ Model berhasil dimuat dari `{model_file}`")

# -----------------------
# Helper: build input frame matching model's original columns
# -----------------------
def build_input_df(values_dict, original_cols):
    """
    values_dict: dict dari nama 'logical' -> value (contoh: 'USIA'->45)
    original_cols: list nama kolom yang dipakai saat training (expected by preprocessor.transform)
    """
    # if original_cols unavailable, try to use keys directly
    if original_cols is None:
        return pd.DataFrame([values_dict])

    # prepare empty frame with original cols
    row = {c: np.nan for c in original_cols}
    # attempt to map common variants
    for k, v in values_dict.items():
        # direct match
        if k in row:
            row[k] = v
            continue
        # try with common suffixes/prefixes
        if f"{k}_num" in row:
            row[f"{k}_num"] = v
            continue
        if k.upper() in row:
            row[k.upper()] = v
            continue
        # also try loose matching by contains (case-insensitive)
        matched = [c for c in row.keys() if k.lower() in c.lower()]
        if len(matched) == 1:
            row[matched[0]] = v
            continue
        # if multiple matches, skip (don't overwrite ambiguous)
    return pd.DataFrame([row])

# -----------------------
# FORM INPUT PASIEN
# -----------------------
st.header("📋 Masukkan Data Pasien")

col1, col2 = st.columns(2)

with col1:
    usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=45)
    tekanan_darah = st.number_input("Tekanan Darah (mmHg) — (systolic)", min_value=60, max_value=300, value=120)
    insulin = st.number_input("Kadar Insulin", min_value=0.0, max_value=500.0, value=85.0)
    glukosa_puasa = st.number_input("Glukosa Darah Puasa (mg/dL)", min_value=30.0, max_value=500.0, value=110.0)
    hbA1c = st.number_input("Persentase HbA1c (%)", min_value=3.0, max_value=20.0, value=6.0)

with col2:
    kolesterol = st.number_input("Kolesterol (mg/dL)", min_value=50.0, max_value=500.0, value=180.0)
    glukosa_sewaktu = st.number_input("Glukosa Darah Sewaktu (mg/dL)", min_value=30.0, max_value=500.0, value=150.0)
    bbtb = st.number_input("BB/TB (BMI)", min_value=10.0, max_value=60.0, value=23.0)
    pola_makan = st.selectbox("Frekuensi Makan per Hari", ["1x", "2x", "3x", "4x", "5x"])
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["LAKI-LAKI", "PEREMPUAN"])
    bbtb_kat = st.selectbox("Kategori BB/TB", ["NORMAL", "OVERWEIGHT", "OBESITAS"])

threshold = st.slider("Ambang Deteksi DM (%)", 0, 100, 60)

st.markdown("---")

# -----------------------
# PREDIKSI BUTTON
# -----------------------
if st.button("🔍 Prediksi Sekarang"):
    # build a logical values dict - these keys will be matched to training columns
    values = {
        # common numeric features (logical names)
        "USIA": usia,
        "Tekanan Darah (mmHg)": tekanan_darah,
        "INSULIN": insulin,
        "Glukosa Darah  Puasa (mg/dL)": glukosa_puasa,
        "Persentase kadar HbA1c (%)": hbA1c,
        "Kolesterol (mg/dL)": kolesterol,
        "Glukosa Darah  Sewaktu (mg/dL)": glukosa_sewaktu,
        "BB/TB": bbtb,
        # categorical logical names
        "POLA MAKAN": pola_makan,
        "JENIS KELAMIN": jenis_kelamin,
        # some datasets may expect a separate BB/TB category column name or BMI category
        "BB/TB_kategori": bbtb_kat,
        "BB/TB_CATEGORY": bbtb_kat
    }

    # if training used BP_SYSTOLIC/BP_DIASTOLIC, try to fill them too
    if original_cols is not None and any("BP_SYSTOLIC" in c for c in original_cols):
        values["BP_SYSTOLIC"] = tekanan_darah
    if original_cols is not None and any("BP_DIASTOLIC" in c for c in original_cols):
        # we cannot reliably get diastolic from single input; set to NaN so imputer handles it,
        # or set equal to systolic * 0.66 as heuristic — here we set NaN to be safe.
        values["BP_DIASTOLIC"] = np.nan

    # Build input DataFrame aligned to original cols
    X_input = build_input_df(values, original_cols)

    # Some preprocessors expect columns names uppercase / specific; ensure strings for categoricals
    # Convert object dtype columns to string for safe processing
    for c in X_input.columns:
        if X_input[c].dtype == object:
            X_input[c] = X_input[c].astype(str)

    # Debug: show prepared input (optional)
    st.write("Input data yang dipersiapkan untuk model:")
    st.dataframe(X_input.T)

    # perform prediction
    try:
        X_proc = preprocessor.transform(X_input)
        prob = model.predict_proba(X_proc)[0][1] if hasattr(model, "predict_proba") else model.predict(X_proc)[0]
        pred = 1 if prob >= (threshold / 100) else 0

        st.subheader("🩸 Hasil Prediksi:")
        st.metric(label="Kemungkinan Diabetes (%)", value=f"{prob*100:.2f}%")

        if pred == 1:
            st.error(f"💀 Hasil: **Positif DM** (≥ {threshold}%)")
        else:
            st.success(f"💚 Hasil: **Negatif / Tidak Terindikasi DM** (< {threshold}%)")

        st.caption(f"Model: Random Forest | Sumber model: `{model_file}` | Validasi: 5-Fold CV (training)")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
        # show traceback-like hint (not full traceback)
        st.write("Periksa apakah kolom input sudah sesuai dengan kolom yang diharapkan model.")
