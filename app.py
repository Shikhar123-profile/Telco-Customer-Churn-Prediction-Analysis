import zipfile
from pathlib import Path
import streamlit as st
import joblib

BASE_DIR = Path(__file__).resolve().parent

with zipfile.ZipFile(BASE_DIR / "churn_scaler (1).zip", "r") as z:
    z.extractall(BASE_DIR)

with zipfile.ZipFile(BASE_DIR / "churn_rf_model (2).zip", "r") as z:
    z.extractall(BASE_DIR)

scaler = joblib.load(BASE_DIR / "churn_scaler.joblib")
model = joblib.load(BASE_DIR / "churn_rf_model.joblib")
