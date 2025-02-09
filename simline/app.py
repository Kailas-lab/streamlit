import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_iris

# Streamlit app title
st.title("Linear Regression App")

# Load Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

st.write("### Data Preview")
st.write(df.head())

# Select features and target
features = st.multiselect("Select feature columns", df.columns[:-1])
target = st.selectbox("Select target column", df.columns)

if features and target:
    X = df[features]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Display model performance
    st.write("### Model Performance")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
    
    # Display regression coefficients
    st.write("### Model Coefficients")
    st.write(pd.DataFrame({"Feature": features, "Coefficient": model.coef_}))
    
    # Plot results if single feature selected
    if len(features) == 1:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test, y_test, color='blue', label='Actual')
        plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
        plt.xlabel(features[0])
        plt.ylabel(target)
        plt.legend()
        st.pyplot(plt)
