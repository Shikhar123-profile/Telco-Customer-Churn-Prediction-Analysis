import zipfile
from pathlib import Path
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide")

@st.cache_resource
def load_artifacts():
    base_dir = Path(__file__).resolve().parent
    with zipfile.ZipFile(base_dir / "churn_scaler (1).zip", "r") as z:
        z.extractall(base_dir)
    with zipfile.ZipFile(base_dir / "churn_rf_model (2).zip", "r") as z:
        z.extractall(base_dir)

    scaler_path = next(base_dir.rglob("*scaler*.joblib"))
    model_path = next(base_dir.rglob("*model*.joblib"))
    return joblib.load(scaler_path), joblib.load(model_path)

scaler, model = load_artifacts()

st.title("Telco Customer Churn Prediction")
st.write("Enter customer usage and plan details below to predict churn risk.")

US_STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL",
    "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA",
    "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
    "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"
]

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Account & Plans")
    state = st.selectbox("State", US_STATES, index=34)  # Default NY
    area_code = st.selectbox("Area Code", [408, 415, 510])
    account_length = st.number_input("Account Length (days)", min_value=1, max_value=300, value=100)
    intl_plan = st.selectbox("International Plan", ["No", "Yes"])
    vmail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
    vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, max_value=60, value=0)
    cust_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=15, value=1)

with col2:
    st.subheader("Day & Evening Usage")
    day_mins = st.number_input("Total Day Minutes", min_value=0.0, max_value=400.0, value=180.0)
    day_calls = st.number_input("Total Day Calls", min_value=0, max_value=200, value=100)
    day_charge = st.number_input("Total Day Charge ($)", min_value=0.0, max_value=70.0, value=30.6)
    eve_mins = st.number_input("Total Evening Minutes", min_value=0.0, max_value=400.0, value=200.0)
    eve_calls = st.number_input("Total Evening Calls", min_value=0, max_value=200, value=100)
    eve_charge = st.number_input("Total Evening Charge ($)", min_value=0.0, max_value=40.0, value=17.0)

with col3:
    st.subheader("Night & International Usage")
    night_mins = st.number_input("Total Night Minutes", min_value=0.0, max_value=400.0, value=200.0)
    night_calls = st.number_input("Total Night Calls", min_value=0, max_value=200, value=100)
    night_charge = st.number_input("Total Night Charge ($)", min_value=0.0, max_value=20.0, value=9.0)
    intl_mins = st.number_input("Total International Minutes", min_value=0.0, max_value=30.0, value=10.0)
    intl_calls = st.number_input("Total International Calls", min_value=0, max_value=25, value=4)
    intl_charge = st.number_input("Total International Charge ($)", min_value=0.0, max_value=10.0, value=2.7)

if st.button("Predict Churn", type="primary"):
    # Initialize all 68 model columns to 0
    row = {col: 0 for col in model.feature_names_in_}

    # Populate raw features
    row["Account length"] = account_length
    row["Area code"] = area_code
    row["International plan"] = 1 if intl_plan == "Yes" else 0
    row["Voice mail plan"] = 1 if vmail_plan == "Yes" else 0
    row["Number vmail messages"] = vmail_messages
    row["Total day minutes"] = day_mins
    row["Total day calls"] = day_calls
    row["Total day charge"] = day_charge
    row["Total eve minutes"] = eve_mins
    row["Total eve calls"] = eve_calls
    row["Total eve charge"] = eve_charge
    row["Total night minutes"] = night_mins
    row["Total night calls"] = night_calls
    row["Total night charge"] = night_charge
    row["Total intl minutes"] = intl_mins
    row["Total intl calls"] = intl_calls
    row["Total intl charge"] = intl_charge
    row["Customer service calls"] = cust_service_calls

    # Set one-hot encoded state column if present (AK is the dropped baseline)
    state_col = f"State_{state}"
    if state_col in row:
        row[state_col] = 1

    input_df = pd.DataFrame([row])[list(model.feature_names_in_)]

    # Scale only the 15 numerical columns expected by scaler
    num_cols = list(scaler.feature_names_in_)
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prediction = model.predict(input_df)[0]
    churn_prob = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"High Churn Risk — Customer is likely to churn (Probability: {churn_prob:.1%})")
    else:
        st.success(f"Low Churn Risk — Customer is likely to stay (Churn Probability: {churn_prob:.1%})")
