# DEPLOYMENT

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained models
logistic_model = joblib.load("logistic_model.pkl")
xgboost_model = joblib.load("tuned_xgboost_model.pkl")
gb_model = joblib.load("tuned_gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")  # Load the scaler

# Streamlit UI
st.title("Telecom Churn Prediction App")
st.write("Enter feature values to predict churn.")

# Model selection
model_choice = st.selectbox("Choose a model:", 
                            ["Logistic Regression", "Tuned XGBoost", "Tuned Gradient Boosting (Highest Accuracy)"])

# Feature selection
all_features = ['day.mins', 'customer.calls', 'eve.mins', 'voice.plan', 'night.mins',
                'account.length', 'intl.mins', 'night.calls', 'day.calls', 'eve.calls']
selected_features = st.multiselect("Select features to include in prediction:", all_features, default=all_features)

# User input fields
user_inputs = {}
if 'day.mins' in selected_features:
    user_inputs['day.mins'] = st.number_input("Day Minutes", value=150.0, format="%.2f")
if 'customer.calls' in selected_features:
    user_inputs['customer.calls'] = st.number_input("Customer Calls", value=5, step=1)
if 'eve.mins' in selected_features:
    user_inputs['eve.mins'] = st.number_input("Evening Minutes", value=150.0, format="%.2f")
if 'voice.plan' in selected_features:
    voice_plan = st.radio("Voice Plan", options=['No', 'Yes'], index=0)
    user_inputs['voice.plan'] = {'No': 0, 'Yes': 1}[voice_plan]
if 'night.mins' in selected_features:
    user_inputs['night.mins'] = st.number_input("Night Minutes", value=150.0, format="%.2f")
if 'account.length' in selected_features:
    user_inputs['account.length'] = st.number_input("Account Length", value=100, step=1)
if 'intl.mins' in selected_features:
    user_inputs['intl.mins'] = st.number_input("International Minutes", value=10.0, format="%.2f")
if 'night.calls' in selected_features:
    user_inputs['night.calls'] = st.number_input("Night Calls", value=75, step=1)
if 'day.calls' in selected_features:
    user_inputs['day.calls'] = st.number_input("Day Calls", value=75, step=1)
if 'eve.calls' in selected_features:
    user_inputs['eve.calls'] = st.number_input("Evening Calls", value=75, step=1)

# Convert user input to DataFrame
input_data = pd.DataFrame([user_inputs])

# Standardize input data
input_data_scaled = scaler.transform(input_data)

# Ensure correct input shape
input_data_scaled = np.array(input_data_scaled).reshape(1, -1)

# Prediction
if st.button("Predict Churn"):
    if model_choice == "Logistic Regression":
        model = logistic_model
    elif model_choice == "Tuned XGBoost":
        model = xgboost_model
    elif model_choice == "Tuned Gradient Boosting (Highest Accuracy)":
        model = gb_model
    
    prediction = model.predict(input_data_scaled)
    st.success(f"The predicted churn is {prediction[0]}")
