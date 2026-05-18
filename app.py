import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open('linear_regression_model.pkl', 'rb'))

# Page Configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 California House Price Predictor")
st.write("Predict house prices using Linear Regression")

st.markdown("---")

# User Inputs
medinc = st.number_input("Median Income", value=3.0)
houseage = st.number_input("House Age", value=20.0)
averooms = st.number_input("Average Rooms", value=5.0)
avebedrms = st.number_input("Average Bedrooms", value=1.0)
population = st.number_input("Population", value=1000.0)
aveoccup = st.number_input("Average Occupancy", value=3.0)
latitude = st.number_input("Latitude", value=34.0)
longitude = st.number_input("Longitude", value=-118.0)

st.markdown("---")

# Prediction Button
if st.button("Predict Price"):

    # Create feature array
    features = np.array([[medinc,
                          houseage,
                          averooms,
                          avebedrms,
                          population,
                          aveoccup,
                          latitude,
                          longitude]])

    # Predict
    prediction = model.predict(features)

    # Display Result
    st.success(f"Predicted House Price: ${prediction[0] * 100000:.2f}")

    st.balloons()

# Footer
st.markdown("---")
st.caption("Machine Learning House Price Prediction using Linear Regression")