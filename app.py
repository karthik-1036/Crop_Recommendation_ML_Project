# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

# App title
st.title("🌾 Crop Recommendation System")

st.markdown("""
This app predicts the **best crop to grow** based on your soil and environmental conditions.
""")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=50)
K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=50)
temperature = st.slider("Temperature (°C)", 10.0, 45.0, 25.0)
humidity = st.slider("Humidity (%)", 10.0, 100.0, 60.0)
ph = st.slider("Soil pH", 3.5, 9.5, 6.5)
rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)

# Predict button
if st.button("Predict Crop"):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(input_data)[0]
    st.success(f"🌱 Recommended Crop: **{prediction}**")
