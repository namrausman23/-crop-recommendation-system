import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- CROP ID TO NAME MAPPING ---
# This dictionary maps the model's numerical output back to the crop name.
# NOTE: This mapping must exactly match the label encoding used during training.
crop_dict = {
    1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coffee', 
    6: 'Kidney Beans', 7: 'Pigeon Peas', 8: 'Moth Beans', 9: 'Mung Bean', 
    10: 'Blackgram', 11: 'Lentil', 12: 'Pomegranate', 13: 'Banana', 
    14: 'Mango', 15: 'Grapes', 16: 'Watermelon', 17: 'Muskmelon', 
    18: 'Apple', 19: 'Orange', 20: 'Papaya', 21: 'Coconut', 22: 'Chickpea'
}

# 1. Load the pre-trained model and scalers
try:
    model = pickle.load(open('model.pkl', 'rb'))
    # Load the scalers used during model training
    scaler = pickle.load(open('standscaler.pkl', 'rb'))
    minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files (pkl) not found. Please ensure they are in the root directory.")
    st.stop()


st.title("🌱 Crop Recommendation System")
st.markdown("Enter the soil and climate parameters below to get the best crop recommendation.")

# 2. Collect Inputs
N = st.number_input('Nitrogen (N)', min_value=0.0, max_value=140.0, value=90.0)
P = st.number_input('Phosphorus (P)', min_value=0.0, max_value=145.0, value=42.0)
K = st.number_input('Potassium (K)', min_value=0.0, max_value=205.0, value=43.0)
temp = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=75.0, step=0.1)
ph = st.number_input('pH Value', min_value=0.0, max_value=14.0, value=6.5, step=0.1)
rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=300.0, value=150.0, step=0.1)

# 3. Prediction Logic
if st.button('Get Recommendation'):
    # Create the input array
    features = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    # Scaling the input data (Must be in the same order as training)
    scaled_features = scaler.transform(features)
    final_features = minmax_scaler.transform(scaled_features)

    # Make prediction (Output is a numerical ID, e.g., 11)
    prediction_id = model.predict(final_features)[0]
    
    # Convert prediction ID to integer for dictionary lookup
    prediction_id = int(prediction_id)

    # Map the numerical ID back to the crop name
    if prediction_id in crop_dict:
        predicted_crop_name = crop_dict[prediction_id]
    else:
        predicted_crop_name = f"Unknown ID: {prediction_id}"
    
    # Display Result
    st.success("✅ Recommendation is ready!")
    st.balloons()
    st.metric("Recommended Crop", predicted_crop_name)