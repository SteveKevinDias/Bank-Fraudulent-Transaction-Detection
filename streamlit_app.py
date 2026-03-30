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
    data = joblib.load('best_fraud_model_tuned.pkl')
    return data

model_data = load_model()
model = model_data['model']
threshold = model_data['threshold']

# Sidebar - Settings
st.sidebar.header("Configuration")
gemini_api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password",
    help="Optional – required to unlock the AI Analyst review."
)

if not gemini_api_key:
    st.sidebar.info("🔑 No API key provided. ML prediction & MongoDB logging are still active. AI Analyst review requires a key.")


@st.cache_resource
def get_mongo_client():
    load_dotenv(override=True)
    mongo_uri = st.secrets["MONGO_URI"]
    if not mongo_uri:
        return None
    try:
        client = MongoClient(mongo_uri)
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
st.markdown(
    "Enter transaction details below to evaluate the likelihood of fraud using our ML model. "
    "Provide a Gemini API key to also receive an AI Analyst review. All results are recorded in MongoDB."
)

# Create columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Details")
    step = st.number_input(
        "Time Step (Hours)", min_value=1, value=1, step=1,
        help="Maps a unit of time in the real world. 1 step is 1 hour of time."
    )
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

    spinner_msg = (
        "Analyzing patterns and consulting AI Analyst..."
        if gemini_api_key
        else "Running ML analysis..."
    )

    with st.spinner(spinner_msg):

        # ------------------------------------------------------------------
        # 1. Feature Engineering
        # ------------------------------------------------------------------
        log_amount = np.log1p(amount)
        is_high_amount = 1 if amount > 1615979.47 else 0   # 99th-percentile from dataset
        hour = step % 24
        is_night = 1 if hour in [0, 1, 2, 3, 4, 5, 22, 23] else 0
        balance_diff_orig = oldbalanceOrg - newbalanceOrig
        balance_diff_dest = newbalanceDest - oldbalanceDest

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

        # ------------------------------------------------------------------
        # 2. ML Prediction  (always runs)
        # ------------------------------------------------------------------
        pos_proba = model.predict_proba(input_data)[0][1]
        model_prediction = "Fraud" if pos_proba >= threshold else "Safe"

        # ------------------------------------------------------------------
        # 3. AI Analysis  (only if API key provided)
        # ------------------------------------------------------------------
        ai_review = None  # None signals "not attempted"

        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
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
                ai_review = f"⚠️ Error calling Gemini AI: {e}"

        # ------------------------------------------------------------------
        # 4. Log to MongoDB  (always runs when connected)
        # ------------------------------------------------------------------
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
                    "confidence": round(float(pos_proba), 6)
                },
                # stored as None/null in MongoDB when no key was provided
                "ai_analysis": ai_review
            }
            collection.insert_one(record)

        # ------------------------------------------------------------------
        # 5. Results Dashboard
        # ------------------------------------------------------------------
        st.markdown("---")
        st.header("Results Dashboard")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.subheader("ML Model Prediction")
            if model_prediction == "Fraud":
                st.error("🚨 **ALERT**: Fraud Detected!")
                st.markdown(f"**Confidence:** {pos_proba:.1%}")
            else:
                st.success("✅ **SAFE**: Legitimate Transaction.")
                st.markdown(f"**Fraud Probability:** {pos_proba:.1%}")

            st.progress(pos_proba, text="Fraud Probability Map")

            if client:
                st.info("💾 Record successfully logged to MongoDB `fraud_detection_db.transactions`.")

        with res_col2:
            st.subheader("🤖 AI Analyst Review")

            if ai_review is None:
                # No key was provided – show a clear, friendly callout
                st.warning(
                    "**AI Analysis requires a Gemini API key.**\n\n"
                    "The ML model has run and results have been saved to MongoDB. "
                    "To unlock the full AI Analyst report, enter your Gemini API key in the sidebar and re-run the analysis."
                )
            else:
                st.markdown(ai_review)
