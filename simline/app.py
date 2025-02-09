import streamlit as st
import numpy as np
import os
import joblib

model = joblib.load('n.joblib')
# Load the trained model


# ---- Streamlit UI Styling ----
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #004085;
            color: #333;
        }
        .main-title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #007BFF;
        }
        .prediction-box {
            background-color: #E3F2FD;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #004085;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- App Layout ----
st.markdown('<h1 class="main-title">💼 Salary Prediction App 📈</h1>', unsafe_allow_html=True)

st.write("### Predict your estimated salary based on years of experience!")

# Sidebar for additional information
st.sidebar.header("ℹ️ About")
st.sidebar.info("This app predicts the salary based on the years of experience using a trained Machine Learning model.")

# Input for years of experience
years_exp = st.slider("Select Your Years of Experience:", min_value=0.0, max_value=50.0, step=0.1, value=2.0)

# Additional input fields for better interactivity
st.write("### Select Additional Options")
gender = st.radio("Select Your Gender:", ("Male", "Female"))
job_type = st.selectbox("Choose Job Type:", ["Software Engineer", "Data Scientist", "Manager", "Analyst", "Other"])

# Prediction Button
if st.button("🚀 Predict Salary"):
    prediction = model.predict(np.array([[years_exp]]))
    
    st.markdown(
        f'<div class="prediction-box">Predicted Salary: ₹{prediction[0]:,.2f}</div>',
        unsafe_allow_html=True
    )

    st.success("✅ Prediction Successful! ")

# Footer
st.markdown("---")
st.caption("📌 Built with ❤️ using Streamlit | Trained with Machine Learning | © 2025")
