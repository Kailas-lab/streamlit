import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set Streamlit page config
st.set_page_config(page_title="Simple Linear Regression App", layout="wide")

# Apply dark theme styling
st.markdown("""
    <style>
    body { background-color: #121212; color: white; }
    .stButton>button { background-color: #BB86FC; color: white; border-radius: 8px; padding: 10px; }
    .sidebar .sidebar-content { background-color: #222; }
    .stTextInput>div>div>input { background-color: #333; color: white; }
    .stDataFrame { background-color: #1E1E1E; color: white; }
    h1, h2, h3, h4 { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.title("📊 Simple Linear Regression App")
st.subheader("Predicting Salaries Based on Years of Experience")

# Dataset Explanation
st.markdown("""
This app uses a **Simple Linear Regression model** to predict salaries based on years of experience.
The dataset should contain two columns:
- **YearsExperience** (Independent Variable)
- **Salary** (Dependent Variable)
""")

# File Upload Section
st.sidebar.header("📂 Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Default dataset path
data_file = "salary_data.csv"

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
elif data_file:
    try:
        data = pd.read_csv(data_file)
        st.success("✅ Using the default dataset (salary_data.csv)")
    except FileNotFoundError:
        st.error("⚠️ No dataset found! Please upload a CSV file.")
        st.stop()
else:
    st.warning("⚠️ Please upload the salary_data.csv file.")
    st.stop()

# Display Dataset Preview
st.write("### 📋 Dataset Preview")
st.dataframe(data.style.set_properties(**{"background-color": "#333", "color": "white"}))

# Ensure required columns exist
if 'YearsExperience' in data.columns and 'Salary' in data.columns:
    # Prepare independent and dependent variables
    X = data[['YearsExperience']].values
    y = data['Salary'].values
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Prediction Section
    st.sidebar.header("🔮 Make a Prediction")
    exp_input = st.sidebar.slider("Enter Years of Experience", min_value=0.0, max_value=50.0, step=0.1)
    predict_button = st.sidebar.button("🚀 Predict Salary", key="predict_button")

    if predict_button:
        prediction = model.predict(np.array([[exp_input]]))
        st.sidebar.success(f"Predicted Salary: **${prediction[0]:,.2f}**")

    # Model Performance Metrics
    st.write("### 📊 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{r2:.2f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("MSE", f"{mse:.2f}")
    col4.metric("RMSE", f"{rmse:.2f}")

    # Actual vs Predicted Line Chart
    st.write("### 📈 Actual vs Predicted Salaries")
    fig, ax = plt.subplots(facecolor='#121212')
    ax.plot(y_test, label='Actual', marker='o', color='cyan')
    ax.plot(y_pred, label='Predicted', marker='x', color='magenta')
    ax.set_xlabel("Test Sample Index", color='white')
    ax.set_ylabel("Salary", color='white')
    ax.legend()
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(colors='white')
    st.pyplot(fig)

    # Additional insights
    if st.button("🔍 Show Model Insights"):
        st.markdown("""
        - **R² Score:** Measures how well the model explains variance in the data.
        - **Lower MAE and RMSE:** Indicate better model performance.
        - **MSE:** Penalizes larger errors more than MAE.
        """, unsafe_allow_html=True)

    # Footer with copyright
    st.markdown("""
    ---
    **© 2025 Simple Linear Regression App | Developed by Kailas M.**
    """, unsafe_allow_html=True)
else:
    st.error("⚠️ Dataset must contain 'YearsExperience' and 'Salary' columns.")
