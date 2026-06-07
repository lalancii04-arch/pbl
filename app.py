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
st.markdown('Masukkan nilai fitur di **sidebar** sebelah kiri, lalu klik **Prediksi**.')
st.divider()

# ---------- Input di sidebar ----------
st.sidebar.header('Input Fitur')
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
        # Susun DataFrame sesuai urutan FITUR
        nilai = pd.DataFrame([[input_user[f] for f in FITUR]], columns=FITUR)
        
        # Standarisasi (Scaling)
        nilai_sc = scaler.transform(nilai)
        
        # Lakukan Prediksi (Menebak Happiness Score)
        pred = model.predict(nilai_sc)[0]

        # Tampilkan hasil (Output diganti dari "usia" menjadi Happiness Score)
        st.success(f'Hasil Prediksi **Happiness Score**: **{pred:,.4f}**')

        # Tampilkan input yang dipakai
        st.subheader('Input yang Digunakan')
        st.dataframe(pd.DataFrame([input_user]), use_container_width=True)

        # Tampilkan koefisien model
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
    st.info('Isi nilai fitur di sidebar, lalu klik tombol Prediksi.')

# ---------- Footer ----------
st.divider()
st.caption('Dibuat untuk PPKD Jakarta Selatan — Kejuruan Data Analyst')