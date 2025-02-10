import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set Streamlit page config
st.set_page_config(page_title="Simple Linear Regression App", layout="wide")

# Apply custom CSS for centering and styling
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stButton>button {
        background-color: #BB86FC;
        color: white;
        border-radius: 8px;
        padding: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #222;
    }
    .stTitle, .stMarkdown, .stHeader, .stDataFrame {
        text-align: center;
    }
    .score-box {
        background-color: #333;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("<h1 style='text-align: center;'>📊 Simple Linear Regression App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This app predicts salaries based on years of experience using a simple linear regression model.</p>", unsafe_allow_html=True)

# Upload dataset
st.markdown("<h3 style='text-align: center;'>📂 Upload Your Dataset (CSV)</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv"])

# Explain dataset format
st.markdown("""
    **📌 Expected Dataset Format:**
    - Column 1: **YearsExperience** (float)
    - Column 2: **Salary** (float)
    - Example:
    
    | YearsExperience | Salary  |
    |----------------|---------|
    | 1.1           | 39343   |
    | 2.0           | 45000   |
    | 3.2           | 60000   |
""", unsafe_allow_html=True)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.markdown("<h3 style='text-align: center;'>📋 Dataset Preview</h3>", unsafe_allow_html=True)
    st.dataframe(data.style.set_properties(**{"background-color": "#333", "color": "white"}))
    
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

        st.markdown("<br><br>", unsafe_allow_html=True)  # Spacing

        # Model performance section
        st.markdown("<h3 style='text-align: center;'>📊 Model Performance Metrics</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='score-box'><b>R² Score</b><br>{r2:.2f}</div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='score-box'><b>MAE</b><br>{mae:.2f}</div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='score-box'><b>MSE</b><br>{mse:.2f}</div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='score-box'><b>RMSE</b><br>{rmse:.2f}</div>", unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)  # Spacing

        # Actual vs Predicted Line Chart
        st.markdown("<h3 style='text-align: center;'>📈 Actual vs Predicted Salaries</h3>", unsafe_allow_html=True)
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

        st.markdown("<br><br>", unsafe_allow_html=True)  # Spacing

        # Prediction Section
        st.sidebar.header("🔮 Make a Prediction")
        exp_input = st.sidebar.slider("Enter Years of Experience", min_value=0.0, max_value=50.0, step=0.1)
        predict_button = st.sidebar.button("Predict Salary", key="predict_button")

        if predict_button:
            prediction = model.predict(np.array([[exp_input]]))
            st.sidebar.success(f"Predicted Salary: ${prediction[0]:,.2f}")

        st.markdown("<br><br>", unsafe_allow_html=True)  # Spacing

        # Insights Button with Working Toggle
        if "show_insights" not in st.session_state:
            st.session_state.show_insights = False

        if st.button("🔍 Show Model Insights"):
            st.session_state.show_insights = not st.session_state.show_insights

        if st.session_state.show_insights:
            st.markdown("""
            **🔍 Model Insights:**
            - **R² Score:** Measures how well the model explains variance in the data.
            - **Lower MAE and RMSE:** Indicate better model performance.
            - **MSE:** Penalizes larger errors more than MAE.
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)  # Spacing

        # Footer with copyright
        st.markdown("""
        <br><br>
        <div style="text-align: center;">
            <b>© 2025 Simple Linear Regression App | Developed by Kailas M.</b>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.error("⚠️ Dataset must contain 'YearsExperience' and 'Salary' columns.")

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
