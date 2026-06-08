import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ---------- Konfigurasi halaman ----------
st.set_page_config(
    page_title='Prediksi Happiness Score',
    page_icon='📊',
    layout='centered',
)

# ---------- Muat model & scaler (cached) ----------
@st.cache_resource
def load_artefak():
    # Menggunakan os.path agar aman dari error "File Not Found" saat deploy
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    model  = joblib.load(os.path.join(BASE_DIR, 'regresi_berganda.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    fitur  = joblib.load(os.path.join(BASE_DIR, 'fitur.pkl'))
    return model, scaler, fitur

model, scaler, FITUR = load_artefak()

# ---------- Header ----------
st.title('📊 Web Prediksi Happiness Score')
st.markdown('Masukkan informasi negara dan nilai fitur di **sidebar** sebelah kiri, lalu klik **Prediksi**.')
st.divider()

# ---------- Input di sidebar ----------
st.sidebar.header('Informasi Data')
# Input tambahan yang HANYA untuk tampilan (tidak masuk ke model)
input_country = st.sidebar.text_input(label='Country (Negara)', value='Indonesia')
input_year = st.sidebar.number_input(label='Year (Tahun)', min_value=2000, max_value=2100, value=2024, step=1)

st.sidebar.divider()

st.sidebar.header('Input Fitur Model')
input_user = {}

# Looping untuk membuat kolom input otomatis sesuai dengan fitur.pkl Anda
for f in FITUR:
    input_user[f] = st.sidebar.number_input(
        label=f,
        value=0.0000,
        step=0.1000,
        format='%.4f',
    )

# ---------- Tombol prediksi ----------
if st.sidebar.button('Prediksi', type='primary', use_container_width=True):
    try:
        # 1. Susun DataFrame KHUSUS FITUR MODEL (Tanpa Country & Year agar tidak error)
        nilai_model = pd.DataFrame([[input_user[f] for f in FITUR]], columns=FITUR)
        
        # 2. Standarisasi (Scaling) hanya untuk fitur numerik
        nilai_sc = scaler.transform(nilai_model)
        
        # 3. Lakukan Prediksi
        pred = model.predict(nilai_sc)[0]

        # 4. Tampilkan hasil (Lebih interaktif dengan menyebutkan nama negara dan tahun)
        st.success(f'Hasil Prediksi **Happiness Score** untuk **{input_country}** tahun **{input_year}**: **{pred:,.4f}**')

        # 5. Siapkan Data Lengkap (Country + Year + Fitur) untuk DITAMPILKAN di tabel UI
        data_tampilan = {
            'Country': input_country,
            'Year': input_year
        }
        data_tampilan.update(input_user) # Menggabungkan input form negara dengan input nilai fitur

        st.subheader('Input yang Digunakan')
        st.dataframe(pd.DataFrame([data_tampilan]), use_container_width=True)

        # 6. Tampilkan koefisien model
        st.subheader('Koefisien Model (Terstandarisasi)')
        df_koef = pd.DataFrame({
            'Fitur': FITUR,
            'Koefisien': model.coef_.round(4),
        })
        st.dataframe(df_koef, use_container_width=True, hide_index=True)
        st.caption(f'Intercept (β₀) = {model.intercept_:.4f}')

    except Exception as e:
        st.error(f'Terjadi error saat memproses prediksi: {e}')
else:
    st.info('Isi informasi negara dan nilai fitur di sidebar, lalu klik tombol Prediksi.')

# ---------- Footer ----------
st.divider()
st.caption('Dibuat untuk Project Learning Based — Kejuruan Data Analyst')
