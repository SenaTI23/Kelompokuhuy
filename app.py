import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Styling custom CSS untuk warna background dan font
st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;
        color: #0a3d62;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1 {
        color: #1e3799;
    }
    .sidebar .sidebar-content {
        background-color: #d6e6f2;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏡 Prediksi Harga Rumah California")
st.write("""
Aplikasi prediksi harga rumah menggunakan **Random Forest Regression**  
Masukkan nilai fitur di sidebar, lalu lihat prediksi harga rumahnya secara real-time.
""")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('rf_california_model_small.pkl')

model = load_model()

# Fitur dari dataset California Housing
feature_names = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

# Sidebar input interaktif dengan deskripsi
st.sidebar.header("🛠️ Input Fitur Rumah")
st.sidebar.markdown("Atur nilai fitur berikut untuk memprediksi harga rumah:")

input_data = {}

input_data['MedInc'] = st.sidebar.slider(
    label='Median Income (10k USD)',
    min_value=0.5, max_value=15.0, value=3.0, step=0.1,
    help='Pendapatan median per rumah dalam satuan 10 ribu USD.'
)
st.sidebar.caption("Nilai ini menunjukkan rata-rata pendapatan rumah tangga.")

input_data['HouseAge'] = st.sidebar.slider(
    label='House Age (tahun)',
    min_value=1, max_value=52, value=20, step=1,
    help='Usia rata-rata rumah dalam tahun.'
)
st.sidebar.caption("Semakin tua rumah, harga bisa berbeda.")

input_data['AveRooms'] = st.sidebar.slider(
    label='Average Rooms per House',
    min_value=1.0, max_value=10.0, value=5.0, step=0.1,
    help='Rata-rata jumlah kamar dalam rumah.'
)
st.sidebar.caption("Termasuk ruang tidur, tamu, dapur, dll.")

input_data['AveBedrms'] = st.sidebar.slider(
    label='Average Bedrooms per House',
    min_value=0.5, max_value=5.0, value=1.0, step=0.1,
    help='Rata-rata jumlah kamar tidur per rumah.'
)
st.sidebar.caption("Berapa kamar tidur yang tersedia.")

input_data['Population'] = st.sidebar.slider(
    label='Population per Block',
    min_value=3, max_value=4000, value=1000, step=1,
    help='Jumlah penduduk di sekitar blok rumah.'
)
st.sidebar.caption("Kepadatan penduduk memengaruhi harga.")

input_data['AveOccup'] = st.sidebar.slider(
    label='Average Occupancy',
    min_value=0.5, max_value=10.0, value=3.0, step=0.1,
    help='Rata-rata jumlah penghuni per rumah.'
)
st.sidebar.caption("Berapa orang biasanya tinggal di rumah tersebut.")

input_data['Latitude'] = st.sidebar.slider(
    label='Latitude',
    min_value=32.5, max_value=42.0, value=34.0, step=0.01,
    help='Letak lintang geografis rumah.'
)
st.sidebar.caption("Lokasi utara-selatan.")

input_data['Longitude'] = st.sidebar.slider(
    label='Longitude',
    min_value=-124.5, max_value=-114.3, value=-118.5, step=0.01,
    help='Letak bujur geografis rumah.'
)
st.sidebar.caption("Lokasi timur-barat.")

# Buat dataframe input untuk model
input_df = pd.DataFrame([input_data])

# Prediksi dalam Rupiah
kurs_usd_to_idr = 15000  # nilai tukar USD ke Rupiah, sesuaikan jika perlu
prediction_usd = model.predict(input_df)[0]
prediction_idr = prediction_usd * 100000 * kurs_usd_to_idr

st.markdown("---")
st.subheader("Prediksi Harga Rumah")
st.markdown(f"<h2 style='color:#079992;'>Rp {prediction_idr:,.0f} </h2>", unsafe_allow_html=True)
st.write("Harga prediksi dalam Rupiah berdasarkan fitur yang kamu masukkan.")

# Tampilkan fitur penting model
st.markdown("---")
st.subheader("Feature Importance Model")

importances = model.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

st.bar_chart(feat_imp)

st.write("Fitur paling berpengaruh pada prediksi harga rumah.")

# Footer cantik
st.markdown("""
    <hr>
    <center>
    <p style='font-size:0.8rem;color:#6c757d;'>Created by SenaTI23 | UAS Machine Learning Genap 2024/2025</p>
    </center>
""", unsafe_allow_html=True)
