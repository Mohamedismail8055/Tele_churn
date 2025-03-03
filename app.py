# DEPLOYMENT

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained models
logistic_model = joblib.load("logistic_model.pkl")
xgboost_model = joblib.load("tuned_xgboost_model.pkl")
gb_model = joblib.load("tuned_gradient_boosting_model.pkl")
rand_forest_model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")  # Load the scaler

# Streamlit UI
st.title("Telecom Churn Prediction App")
st.write("Enter feature values to predict churn.")

# Model selection
model_choice = st.selectbox("Choose a model:", 
                            ["Logistic Regression", "Tuned XGBoost", "Tuned Gradient Boosting (Highest Accuracy)", "Random Forest"])

# User input fields (Top 10 Features Only)
input_data = {
    'day.mins': st.slider("Day Minutes", min_value=0.0, max_value=300.0, step=0.1, value=150.0),
    'customer.calls': st.slider("Customer Calls", min_value=0, max_value=20, step=1, value=5),
    'eve.mins': st.slider("Evening Minutes", min_value=0.0, max_value=300.0, step=0.1, value=150.0),
    'voice.plan': st.radio("Voice Plan", options=[0, 1], index=0),
    'night.mins': st.slider("Night Minutes", min_value=0.0, max_value=300.0, step=0.1, value=150.0),
    'account.length': st.slider("Account Length", min_value=1, max_value=243, step=1, value=100),
    'intl.mins': st.slider("International Minutes", min_value=0.0, max_value=20.0, step=0.1, value=10.0),
    'night.calls': st.slider("Night Calls", min_value=0, max_value=160, step=1, value=75),
    'day.calls': st.slider("Day Calls", min_value=0, max_value=160, step=1, value=75),
    'eve.calls': st.slider("Evening Calls", min_value=0, max_value=160, step=1, value=75)
}

# Convert user input to DataFrame
input_data = pd.DataFrame([input_data])

# Ensure input_data matches the training feature order
expected_features = list(scaler.feature_names_in_)  # Get expected feature names from the scaler
input_data = input_data.reindex(columns=expected_features, fill_value=0)  # Reorder and fill missing features

# Standardize input data
input_data_scaled = scaler.fit_transform(input_data)

# Reshape input_data_scaled to ensure compatibility
input_data_scaled = input_data_scaled.reshape(1, -1)  # Ensure correct input shape

# Prediction
if st.button("Predict Churn"):
    if model_choice == "Logistic Regression":
        model = logistic_model
    elif model_choice == "Tuned XGBoost":
        model = xgboost_model
    elif model_choice == "Tuned Gradient Boosting (Highest Accuracy)":
        model = gb_model
    else:
        model = rand_forest_model
    
    prediction = model.predict(input_data_scaled)
    st.success(f"The predicted churn is {prediction[0]}")
