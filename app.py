import zipfile
from pathlib import Path
import streamlit as st
import joblib

BASE_DIR = Path(__file__).resolve().parent

with zipfile.ZipFile(BASE_DIR / "churn_scaler (1).zip", "r") as z:
    z.extractall(BASE_DIR)

with zipfile.ZipFile(BASE_DIR / "churn_rf_model (2).zip", "r") as z:
    z.extractall(BASE_DIR)

scaler_path = next(BASE_DIR.rglob("*scaler*.joblib"))
model_path = next(BASE_DIR.rglob("*model*.joblib"))

scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

st.success("Scaler and model loaded successfully!")
