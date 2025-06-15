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

# Sidebar input interaktif
st.sidebar.header("Input Fitur Rumah")

input_data = {}
input_data['MedInc'] = st.sidebar.slider('Median Income (10k USD)', 0.5, 15.0, 3.0, 0.1)
input_data['HouseAge'] = st.sidebar.slider('House Age (years)', 1, 52, 20)
input_data['AveRooms'] = st.sidebar.slider('Average Rooms per House', 1.0, 10.0, 5.0, 0.1)
input_data['AveBedrms'] = st.sidebar.slider('Average Bedrooms per House', 0.5, 5.0, 1.0, 0.1)
input_data['Population'] = st.sidebar.slider('Population per Block', 3, 4000, 1000)
input_data['AveOccup'] = st.sidebar.slider('Average Occupancy', 0.5, 10.0, 3.0, 0.1)
input_data['Latitude'] = st.sidebar.slider('Latitude', 32.5, 42.0, 34.0, 0.01)
input_data['Longitude'] = st.sidebar.slider('Longitude', -124.5, -114.3, -118.5, 0.01)

# Buat dataframe input untuk model
input_df = pd.DataFrame([input_data])

# Prediksi
prediction = model.predict(input_df)[0]

st.markdown("---")
st.subheader("Prediksi Harga Rumah")
st.markdown(f"<h2 style='color:#079992;'>${prediction * 100000:.2f} USD</h2>", unsafe_allow_html=True)
st.write("Harga prediksi dalam USD berdasarkan fitur yang kamu masukkan.")

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
