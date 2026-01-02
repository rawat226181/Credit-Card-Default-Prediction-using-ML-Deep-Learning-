import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from tensorflow import keras

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Credit Default Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
.main { padding: 2rem; }
.stMetric {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_models():
    with open("credit_default_ml_model.pkl", "rb") as f:
        ml_artifacts = pickle.load(f)

    dnn_model = keras.models.load_model("credit_default_dnn_model.h5")

    with open("credit_default_dnn_artifacts.pkl", "rb") as f:
        dnn_artifacts = pickle.load(f)

    return ml_artifacts, dnn_model, dnn_artifacts


ml_artifacts, dnn_model, dnn_artifacts = load_models()

ml_model = ml_artifacts["model"]
scaler = ml_artifacts["scaler"]
feature_cols = ml_artifacts["feature_cols"]
encoders = ml_artifacts["encoders"]

# =====================================================
# TITLE
# =====================================================
st.title("💳 Credit Card Default Risk Prediction System")
st.markdown("### ML & Deep Learning based Financial Risk Assessment")
st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("⚙️ Settings")

    prediction_mode = st.radio(
        "Prediction Mode",
        ["Single Customer", "Batch Processing"]
    )

    model_choice = st.radio(
        "Select Model",
        ["Machine Learning (Gradient Boosting)", "Deep Neural Network"]
    )

    st.markdown("---")
    st.header("📊 Model Performance")

    if model_choice.startswith("Machine"):
        st.metric("ROC-AUC", f"{ml_artifacts['roc_auc']:.2%}")
    else:
        st.metric("ROC-AUC", f"{dnn_artifacts['roc_auc']:.2%}")

# =====================================================
# SINGLE CUSTOMER MODE
# =====================================================
if prediction_mode == "Single Customer":

    st.header("👤 Single Customer Risk Assessment")

    tab1, tab2, tab3 = st.tabs(["Basic Info", "Financial Details", "Payment History"])

    with tab1:
        age = st.number_input("Age", 18, 100, 35)
        gender = st.selectbox("Gender", ["M", "F"])
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        employment_status = st.selectbox("Employment", ["Employed", "Self-Employed", "Unemployed"])
        num_dependents = st.number_input("Dependents", 0, 10, 0)

    with tab2:
        credit_limit = st.number_input("Credit Limit", 1000, 100000, 20000)
        avg_monthly_income = st.number_input("Avg Monthly Income", 0, 50000, 5000)
        account_age_months = st.number_input("Account Age (Months)", 1, 240, 24)
        num_bank_accounts = st.number_input("Bank Accounts", 1, 10, 2)
        num_credit_cards = st.number_input("Credit Cards", 1, 10, 3)

        bill_amt_1 = st.number_input("Latest Bill Amount", 0, 50000, 5000)
        pay_amt_1 = st.number_input("Latest Payment Amount", 0, 50000, 1000)

    with tab3:
        pay_status_1 = st.selectbox("Latest Payment Status", [-1, 0, 1], index=1)

    if st.button("🔮 Predict Default Risk", use_container_width=True):

        input_data = {
            "age": age,
            "gender_encoded": encoders["gender"].transform([gender])[0],
            "education_encoded": encoders["education"].transform([education])[0],
            "marital_encoded": encoders["marital"].transform([marital_status])[0],
            "employment_encoded": encoders["employment"].transform([employment_status])[0],
            "num_dependents": num_dependents,
            "credit_limit": credit_limit,
            "account_age_months": account_age_months,
            "num_bank_accounts": num_bank_accounts,
            "num_credit_cards": num_credit_cards,
            "avg_monthly_income": avg_monthly_income,
            "pay_status_1": pay_status_1,
            "pay_status_2": 0,
            "pay_status_3": 0,
            "pay_status_4": 0,
            "pay_status_5": 0,
            "pay_status_6": 0,
            "bill_amt_1": bill_amt_1,
            "bill_amt_2": bill_amt_1,
            "bill_amt_3": bill_amt_1,
            "pay_amt_1": pay_amt_1,
            "pay_amt_2": pay_amt_1,
            "pay_amt_3": pay_amt_1,
        }

        # Engineered features
        input_data["utilization_ratio"] = bill_amt_1 / (credit_limit + 1)
        input_data["avg_utilization"] = input_data["utilization_ratio"]
        input_data["max_utilization"] = input_data["utilization_ratio"]
        input_data["total_delays"] = int(pay_status_1 == -1)
        input_data["recent_delays"] = input_data["total_delays"]
        input_data["payment_trend"] = 0
        input_data["bill_increase"] = 0
        input_data["bill_volatility"] = 0
        input_data["avg_bill"] = bill_amt_1
        input_data["payment_to_bill_ratio"] = pay_amt_1 / (bill_amt_1 + 1)
        input_data["avg_payment"] = pay_amt_1
        input_data["payment_decrease"] = 0
        input_data["debt_to_income"] = bill_amt_1 / (avg_monthly_income + 1)
        input_data["credit_per_card"] = credit_limit / num_credit_cards
        input_data["income_to_limit"] = avg_monthly_income / (credit_limit + 1)
        input_data["high_utilization"] = int(input_data["utilization_ratio"] > 0.8)
        input_data["consistent_delays"] = int(input_data["total_delays"] > 3)
        input_data["low_payment"] = int(input_data["payment_to_bill_ratio"] < 0.1)
        input_data["young_age"] = int(age < 25)
        input_data["unemployed"] = int(employment_status == "Unemployed")
        input_data["account_maturity_years"] = account_age_months / 12
        input_data["is_new_account"] = int(account_age_months < 12)

        df = pd.DataFrame([input_data])[feature_cols]
        df_scaled = scaler.transform(df)

        if model_choice.startswith("Machine"):
            prob = ml_model.predict_proba(df_scaled)[0][1]
        else:
            prob = dnn_model.predict(df_scaled, verbose=0)[0][0]

        st.success(f"### Default Probability: **{prob*100:.2f}%**")

# =====================================================
# BATCH MODE
# =====================================================
else:
    st.header("📁 Batch Credit Risk Assessment")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        st.dataframe(df_batch.head())

        if st.button("🚀 Run Batch Prediction", use_container_width=True):

            # Encode
            df_batch["gender_encoded"] = encoders["gender"].transform(df_batch["gender"])
            df_batch["education_encoded"] = encoders["education"].transform(df_batch["education"])
            df_batch["marital_encoded"] = encoders["marital"].transform(df_batch["marital_status"])
            df_batch["employment_encoded"] = encoders["employment"].transform(df_batch["employment_status"])

            # Simple engineered features
            df_batch["utilization_ratio"] = df_batch["bill_amt_1"] / (df_batch["credit_limit"] + 1)
            df_batch["payment_to_bill_ratio"] = df_batch["pay_amt_1"] / (df_batch["bill_amt_1"] + 1)
            df_batch["debt_to_income"] = df_batch["bill_amt_1"] / (df_batch["avg_monthly_income"] + 1)
            df_batch["account_maturity_years"] = df_batch["account_age_months"] / 12

            X = scaler.transform(df_batch[feature_cols])

            if model_choice.startswith("Machine"):
                probs = ml_model.predict_proba(X)[:, 1]
            else:
                probs = dnn_model.predict(X, verbose=0).flatten()

            df_batch["default_probability"] = probs
            df_batch["risk_level"] = pd.cut(
                probs, [0, 0.3, 0.7, 1.0], labels=["Low", "Medium", "High"]
            )

            st.success("✅ Batch Prediction Completed")

            fig = px.histogram(df_batch, x="default_probability", title="Default Probability Distribution")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_batch)

            st.download_button(
                "⬇️ Download Results",
                df_batch.to_csv(index=False),
                file_name="credit_default_results.csv",
                mime="text/csv"
            )
