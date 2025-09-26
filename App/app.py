import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, and feature list
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_list = joblib.load('scaled_feature.pkl')

# Ensure feature_list is a 1D flat list
if isinstance(feature_list, (np.ndarray, pd.Index)):
    feature_list = feature_list.tolist()
if len(feature_list) == 1 and isinstance(feature_list[0], (list, np.ndarray, pd.Index)):
    feature_list = list(feature_list[0])

st.title("📊Telco Customer Churn Prediction")
st.markdown("""This app predicts whether a Telco customer is likely to churn (cancel service) based on their account information.Please answer the questions below to get your prediction.""")

# Define questions and input types
question_map = {
    'tenure': "How many months has the customer stayed with the company?",
    'InternetService_Fiber optic': "Does the customer have Fiber optic internet service?",
    'InternetService_No': "Does the customer NOT have internet service?",
    'OnlineSecurity_No internet service': "Does the customer NOT have online security service?",
    'OnlineBackup_No internet service': "Does the customer NOT have online backup service?",
    'DeviceProtection_No internet service': "Does the customer NOT have device protection service?",
    'TechSupport_No internet service': "Does the customer NOT have tech support service?",
    'StreamingTV_No internet service': "Does the customer NOT have streaming TV service?",
    'StreamingMovies_No internet service': "Does the customer NOT have streaming movies service?",
    'Contract_Two year': "Is the customer on a Two year contract?",
    'PaymentMethod_Electronic check': "Is the payment method Electronic check?"
}

# Define feature types
feature_types = {
    'tenure': 'number',
    'InternetService_Fiber optic': 'checkbox',
    'InternetService_No': 'checkbox',
    'OnlineSecurity_No internet service': 'checkbox',
    'OnlineBackup_No internet service': 'checkbox',
    'DeviceProtection_No internet service': 'checkbox',
    'TechSupport_No internet service': 'checkbox',
    'StreamingTV_No internet service': 'checkbox',
    'StreamingMovies_No internet service': 'checkbox',
    'Contract_Two year': 'checkbox',
    'PaymentMethod_Electronic check': 'checkbox'
}



input_data = {}

# Collect user inputs
for feature in feature_list:
    ftype = feature_types.get(feature, 'text')
    question = question_map.get(feature, feature.replace('_', ' ').capitalize())

    if ftype == 'number':
        input_data[feature] = st.number_input(f"{question}", min_value=0)
    elif ftype == 'checkbox':
        input_data[feature] = int(st.checkbox(f"{question}"))
    else:
        input_data[feature] = st.text_input(f"{question}")


# Predict
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        if prediction[0] == 1:
            st.error("❌ Customer is likely to churn!")
            st.write(f"Churn Probability: **{prediction_proba[0][1]:.2%}**")
        else:
            st.success("✅ Customer is unlikely to churn.")
            st.write(f"Churn Probability: **{prediction_proba[0][0]:.2%}**")
       
    except Exception as e:
        import traceback
        st.error(f"⚠️ Error during prediction: {str(e)}")
        st.code(traceback.format_exc())  # Show full stack trace
