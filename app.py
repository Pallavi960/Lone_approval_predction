import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")

# ---- Load model artifacts ----
try:
    with open("loan_approval_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(
        "Could not find loan_approval_model.pkl. "
        "Make sure it's in the same folder as this app.py file."
    )
    st.stop()

# scaler.pkl is optional -- only present if you saved one separately.
# (GradientBoostingClassifier doesn't require scaling, so it's fine if this is missing.)
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = None

# Default column order matches the form fields below.
# If you later save a columns.pkl (recommended, to guarantee exact training order),
# it will override this default.
default_columns = [
    "no_of_dependents", "is_Graduate", "self_employed", "income_annum",
    "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value",
]
try:
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
except FileNotFoundError:
    columns = default_columns

st.title("🏦 Loan Approval Prediction")
st.write("Fill in the applicant details below to check loan approval status.")

# ---- Input form ----
no_of_dependents = st.slider("Number of Dependents", 0, 10, 0)
is_graduate = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income", min_value=0, value=500000, step=10000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=1000000, step=10000)
loan_term = st.slider("Loan Term (Months)", 1, 30, 12)
cibil_score = st.slider("CIBIL Score", 300, 900, 650)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=0, step=10000)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=0, step=10000)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=0, step=10000)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=0, step=10000)

if st.button("Predict", type="primary"):

    input_data = {
        "no_of_dependents": no_of_dependents,
        "is_Graduate": 1 if is_graduate == "Graduate" else 0,
        "self_employed": 1 if self_employed == "Yes" else 0,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }

    input_df = pd.DataFrame([input_data])

    # Reorder columns to exactly match training order
    missing_cols = set(columns) - set(input_df.columns)
    if missing_cols:
        st.error(f"Missing expected columns: {missing_cols}. Check your columns.pkl matches this form.")
        st.stop()
    input_df = input_df[columns]

    # Apply scaler only if one was actually used during training
    if scaler is not None:
        input_df = scaler.transform(input_df)

    prediction = model.predict(input_df)[0]

    # Show probability if the model supports it
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]

    st.divider()
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    if proba is not None:
        st.write(f"Confidence — Approved: **{proba[1]*100:.1f}%**, Rejected: **{proba[0]*100:.1f}%**")