import streamlit as st
import numpy as np
import pandas as pd
from src.pipeline.prediction import CustomerData, PredictionPipeline

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction App")

st.markdown("### Enter Customer Details:")

# Sidebar for inputs
with st.form("churn_form"):
    gender = st.radio("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.radio("Partner", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    Dependents = st.radio("Dependents", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.radio("Phone Service", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    MultipleLines = st.selectbox("Multiple Lines", [1, 0, -1], format_func=lambda x: {1: "Yes", 0: "No", -1: "No phone service"}[x])
    InternetService = st.selectbox("Internet Service", [1, 2, 0], format_func=lambda x: {1: "DSL", 2: "Fiber optic", 0: "No"}[x])
    OnlineSecurity = st.selectbox("Online Security", [1, 0, -1], format_func=lambda x: {1: "Yes", 0: "No", -1: "No internet service"}[x])
    OnlineBackup = st.selectbox("Online Backup", [1, 0, -1], format_func=lambda x: {1: "Yes", 0: "No", -1: "No internet service"}[x])
    DeviceProtection = st.selectbox("Device Protection", [1, 0, -1], format_func=lambda x: {1: "Yes", 0: "No", -1: "No internet service"}[x])
    TechSupport = st.selectbox("Tech Support", [1, 0, -1], format_func=lambda x: {1: "Yes", 0: "No", -1: "No internet service"}[x])
    StreamingTV = st.selectbox("Streaming TV", [1, 0, -1], format_func=lambda x: {1: "Yes", 0: "No", -1: "No internet service"}[x])
    StreamingMovies = st.selectbox("Streaming Movies", [1, 0, -1], format_func=lambda x: {1: "Yes", 0: "No", -1: "No internet service"}[x])
    Contract = st.selectbox("Contract", [0, 1, 2], format_func=lambda x: {0: "Month-to-month", 1: "One year", 2: "Two year"}[x])
    PaperlessBilling = st.radio("Paperless Billing", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    PaymentMethod = st.selectbox("Payment Method", [0, 1, 2], format_func=lambda x: {0: "Mailed check", 1: "Bank transfer (automatic)", 2: "Electronic check"}[x])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, step=1.0)
    
    submit = st.form_submit_button("Predict")

if submit:
    try:
        # Prepare data
        customer_data = CustomerData(
            gender, SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, MultipleLines, InternetService, OnlineSecurity,
            OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
            StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
            MonthlyCharges, TotalCharges
        )
        
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(customer_data)
        
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error("Customer is likely to churn.")
        else:
            st.success("Customer is not likely to churn.")
    
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
