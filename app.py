import streamlit as st
import pandas as pd
import joblib

# Load model yang sudah disimpan
model = model = joblib.load('rf_california_model_small.pkl')

st.title("Prediksi Harga Rumah California dengan Random Forest")

st.sidebar.header("Masukkan Fitur Rumah")

def user_input_features():
    MedInc = st.sidebar.slider('Median Income (MedInc)', 0.5, 15.0, 3.5)
    HouseAge = st.sidebar.slider('House Age (tahun)', 1, 52, 20)
    AveRooms = st.sidebar.slider('Rata-rata jumlah kamar', 0.5, 15.0, 5.0)
    AveBedrms = st.sidebar.slider('Rata-rata kamar tidur', 0.5, 5.0, 1.0)
    Population = st.sidebar.slider('Populasi', 3, 35600, 1000)
    AveOccup = st.sidebar.slider('Rata-rata penghuni per rumah', 0.5, 20.0, 3.0)
    Latitude = st.sidebar.slider('Latitude', 32.5, 42.0, 34.0)
    Longitude = st.sidebar.slider('Longitude', -124.0, -114.0, -118.0)
    
    data = {'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup,
            'Latitude': Latitude,
            'Longitude': Longitude}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('Input Data')
st.write(input_df)

# Prediksi harga rumah
prediction = model.predict(input_df)

st.subheader('Prediksi Harga Rumah (dalam 100,000 USD)')
st.write(f"${prediction[0]*100000:,.2f}")

# Visualisasi sederhana fitur penting (hardcode)
feature_importances = {
    'MedInc': 0.43,
    'Longitude': 0.19,
    'Latitude': 0.13,
    'AveRooms': 0.06,
    'HouseAge': 0.05,
    'AveOccup': 0.04,
    'Population': 0.03,
    'AveBedrms': 0.03
}

st.subheader("Feature Importance")
st.bar_chart(pd.Series(feature_importances))
