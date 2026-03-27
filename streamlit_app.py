import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import google.generativeai as genai
from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os

load_dotenv(override=True)

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Bank Fraud Detection App",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model
@st.cache_resource
def load_model():
    # Load the dictionary
    data = joblib.load('best_fraud_model_tuned.pkl')
    return data

model_data = load_model()
model = model_data['model']
threshold = model_data['threshold']

# Sidebar - Settings
st.sidebar.header("Configuration")
gemini_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Required to generate the AI Analyst review.")


@st.cache_resource
def get_mongo_client():
    load_dotenv(override=True)
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        return None
    try:
        client = MongoClient(mongo_uri)
        # Send a ping to confirm a successful connection
        client.admin.command('ping')
        return client
    except Exception as e:
        st.sidebar.error(f"MongoDB error: {e}")
        return None

client = get_mongo_client()
if client:
    db = client['fraud_detection_db']
    collection = db['transactions']
    st.sidebar.success("MongoDB Connected!")
else:
    st.sidebar.error("Failed to connect to MongoDB.")

# UI Header
st.title("🛡️ Advanced Fraud Detection & AI Analyst")
st.markdown("Enter transaction details below to evaluate the likelihood of fraud using our ML model, audited by a Google Gemini AI Analyst and recorded into MongoDB.")

# Create columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Details")
    step = st.number_input("Time Step (Hours)", min_value=1, value=1, step=1, help="Maps a unit of time in the real world. 1 step is 1 hour of time.")
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150000.0, step=1000.0)

with col2:
    st.subheader("Origin Account")
    oldbalanceOrg = st.number_input("Sender Initial Balance ($)", min_value=0.0, value=200000.0, step=1000.0)
    newbalanceOrig = st.number_input("Sender Final Balance ($)", min_value=0.0, value=50000.0, step=1000.0)

st.subheader("Destination Account")
col3, col4 = st.columns(2)
with col3:
    oldbalanceDest = st.number_input("Receiver Initial Balance ($)", min_value=0.0, value=0.0, step=1000.0)
with col4:
    newbalanceDest = st.number_input("Receiver Final Balance ($)", min_value=0.0, value=150000.0, step=1000.0)

# Prediction Button
if st.button("Analyze Transaction", use_container_width=True, type="primary"):
    if not gemini_api_key:
        st.error("Please provide your Gemini API Key in the sidebar to run the analysis.")
    else:
        with st.spinner("Analyzing patterns and consulting AI Analyst..."):
            # Derive features based on notebook logic
            log_amount = np.log1p(amount)
            # Using 99th percentile determined from dataset: 1615979.47
            is_high_amount = 1 if amount > 1615979.47 else 0
            hour = step % 24
            is_night = 1 if hour in [0, 1, 2, 3, 4, 5, 22, 23] else 0
            balance_diff_orig = oldbalanceOrg - newbalanceOrig
            balance_diff_dest = newbalanceDest - oldbalanceDest
            
            # Build Dataframe mapping exact feature names
            input_data = pd.DataFrame([[
                step, 
                amount, 
                log_amount, 
                is_high_amount, 
                hour, 
                is_night, 
                balance_diff_orig, 
                balance_diff_dest
            ]], columns=model_data['features'])
            
            # Predict
            pos_proba = model.predict_proba(input_data)[0][1]
            model_prediction = "Fraud" if pos_proba >= threshold else "Safe"
            
            # Call Gemini
            genai.configure(api_key=gemini_api_key)
            # Using gemini-2.5-flash for fastest, most reliable generation over textual data
            llm_model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = f"""
            You are a Senior Bank Fraud Analyst. We have a transaction and a Machine Learning model prediction.
            
            Transaction Details:
            - Time Step: {step}
            - Amount: ${amount}
            - Sender Initial Balance: ${oldbalanceOrg}
            - Sender Final Balance: ${newbalanceOrig}
            - Sender Balance Difference: ${balance_diff_orig}
            - Receiver Initial Balance: ${oldbalanceDest}
            - Receiver Final Balance: ${newbalanceDest}
            - Receiver Balance Difference: ${balance_diff_dest}
            
            ML Model Prediction: {model_prediction}
            ML Probability of Fraud: {pos_proba:.1%}
            
            Task:
            1. Evaluate if the transaction genuinely looks suspicious based on the mathematical logic (e.g., if balance difference doesn't match the amount exactly).
            2. Compare your evaluation with the ML Model Prediction.
            3. *Crucial Rule 1*: If you believe the model incorrectly identified the transaction (e.g., the model says 'Safe' but the math shows a massive unexplained wipeout, or vice versa), explicitly evaluate the pros and cons of concluding it is fraud vs safe, and then make a definitive final decision.
            4. *Crucial Rule 2*: If you can't find a direct logical/mathematical explanation for the model's prediction, keep in mind that the ML model may have detected deeper, non-obvious experimental patterns from human behavior (like time-of-day anomalies or specific transferring habits). Even if the reasoning doesn't make perfect sense, weigh the risks based on the model's confidence.
            5. Provide explicit and Final Actionable Next Steps for our operations team based on your ultimate decision.
            
            Return your response in markdown format.
            """
            
            try:
                response = llm_model.generate_content(prompt)
                ai_review = response.text
            except Exception as e:
                ai_review = f"Error calling Gemini AI: {e}"
            
            # Log to MongoDB
            if client:
                record = {
                    "timestamp": datetime.utcnow(),
                    "transaction": {
                        "step": step,
                        "amount": amount,
                        "oldbalanceOrg": oldbalanceOrg,
                        "newbalanceOrig": newbalanceOrig,
                        "oldbalanceDest": oldbalanceDest,
                        "newbalanceDest": newbalanceDest
                    },
                    "derived_features": {
                        "hour": hour,
                        "is_night": is_night,
                        "balance_diff_orig": balance_diff_orig,
                        "balance_diff_dest": balance_diff_dest
                    },
                    "ml_model": {
                        "prediction": model_prediction,
                        "confidence": pos_proba
                    },
                    "llm_review": ai_review
                }
                collection.insert_one(record)
            
            st.markdown("---")
            st.header("Results Dashboard")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.subheader("ML Model Prediction")
                if model_prediction == "Fraud":
                    st.error(f"🚨 **ALERT**: Fraud Detected!")
                    st.markdown(f"**Confidence:** {pos_proba:.1%}")
                else:
                    st.success(f"✅ **SAFE**: Legitimate Transaction.")
                    st.markdown(f"**Fraud Probability:** {pos_proba:.1%}")
                
                st.progress(pos_proba, text="Fraud Probability Map")
                if client:
                    st.info("💾 Record successfully logged to MongoDB `fraud_detection_db.transactions`.")
                    
            with res_col2:
                st.subheader("🤖 AI Analyst Review")
                st.markdown(ai_review)
