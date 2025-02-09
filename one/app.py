import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("n.joblib")

st.title("Salary Prediction App")

# User input
years_experience = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, step=0.1)

if st.button("Predict Salary"):
    input_data = np.array([[years_experience]])  # Convert input to 2D array
    predicted_salary = model.predict(input_data)[0]  # Get the prediction
    st.write(f"Predicted Salary: ${predicted_salary:,.2f}")
