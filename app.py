import streamlit as st
import joblib
scaler = joblib.load("03_models/churn_scaler.joblib")
model = joblib.load("03_models/churn_rf_model.joblib")
