import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set Streamlit page config
st.set_page_config(page_title="Simple Linear Regression App", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: white;
        text-align: center;
    }
    .stButton>button {
        background-color: #BB86FC;
        color: white;
        border-radius: 8px;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #3700B3;
    }
    .sidebar .sidebar-content {
        background-color: #222;
    }
    .metric-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        padding: 10px;
    }
    .metric-box {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        min-width: 150px;
    }
    .centered {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='centered'>📊 Simple Linear Regression on Salary Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='centered'>This app helps predict salaries based on years of experience using a simple linear regression model.</p>", unsafe_allow_html=True)

# Load dataset from the same folder
data_file = "salary_data.csv"
data = pd.read_csv(data_file)

st.markdown("<h2 class='centered'>📋 Dataset Preview</h2>", unsafe_allow_html=True)
st.dataframe(data.style.set_properties(**{"background-color": "#333", "color": "white"}))

# Check if required columns exist
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
    
    st.sidebar.header("🔮 Make a Prediction")
    exp_input = st.sidebar.slider("Enter Years of Experience", min_value=0.0, max_value=50.0, step=0.1)
    predict_button = st.sidebar.button("Predict Salary", key="predict_button")
    
    if predict_button:
        prediction = model.predict(np.array([[exp_input]]))
        st.sidebar.success(f"Predicted Salary: ${prediction[0]:,.2f}")
        st.sidebar.write("💡 More experience generally leads to a higher salary!")
    
    # Show model performance metrics
    st.markdown("<h2 class='centered'>📊 Model Performance Metrics</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class="metric-container">
            <div class="metric-box">
                <h4>R² Score</h4>
                <p>{:.2f}</p>
            </div>
            <div class="metric-box">
                <h4>MAE</h4>
                <p>{:.2f}</p>
            </div>
            <div class="metric-box">
                <h4>MSE</h4>
                <p>{:.2f}</p>
            </div>
            <div class="metric-box">
                <h4>RMSE</h4>
                <p>{:.2f}</p>
            </div>
        </div>
    """.format(r2, mae, mse, rmse), unsafe_allow_html=True)
    
    # Actual vs Predicted Line Chart
    st.markdown("<h2 class='centered'>📈 Actual vs Predicted Salaries</h2>", unsafe_allow_html=True)
    fig, ax = plt.subplots(facecolor='#121212')
    ax.plot(y_test, label='Actual', marker='o', color='cyan')
    ax.plot(y_pred, label='Predicted', marker='x', color='magenta')
    ax.set_xlabel("Test Sample Index", color='white')
    ax.set_ylabel("Salary", color="white")
    ax.legend()
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(colors='white')
    st.pyplot(fig)
    
    # Additional insights
    if st.button("🔍 Show Model Insights"):
        st.markdown("<h3 class='centered'>🔍 Model Insights</h3>", unsafe_allow_html=True)
        st.write("The R² score measures how well the model explains the variance in the data.")
        st.write("Lower MAE and RMSE indicate better model performance.")
        st.write("MSE penalizes larger errors more heavily than MAE.")
        st.info("💡 Tip: Try different test sizes and training data proportions to improve accuracy!")
    
    # Footer with copyright
    st.markdown("""
    ---
    <p class='centered'><b>© 2025 Simple Linear Regression App | Developed by Kailas M.</b></p>
    """, unsafe_allow_html=True)
else:
    st.error("⚠️ Dataset must contain 'YearsExperience' and 'Salary' columns.")
