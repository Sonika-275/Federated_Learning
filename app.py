# app.py
import streamlit as st
import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from model import FraudDetectionModel

st.set_page_config(page_title="Federated Fraud Detection", layout="wide")

st.title("🏦 Federated Learning Fraud Detection Dashboard")

# ── LOAD MODEL ─────────────────────────────────────
def load_model(path):
    model = FraudDetectionModel()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load all models
model_a = load_model("model/local_model_a.pth")
model_b = load_model("model/local_model_b.pth")
model_c = load_model("model/local_model_c.pth")
global_model = load_model("model/global_model.pth")

# ── LOAD DATA ─────────────────────────────────────
def load_random_transaction(bank):
    df = pd.read_csv(f"data/bank_{bank}.csv")
    
    # ✅ Pick only fraud rows
    fraud_df = df[df['Class'] == 1]
    
    # If no fraud rows exist
    if len(fraud_df) == 0:
        st.error("No fraud cases in this dataset!")
        return None, None
    
    # Pick random fraud row
    row = fraud_df.sample(n=1)
    
    # Prepare features
    X = df.drop(['Class', 'Time'], axis=1)
    row_features = row.drop(['Class', 'Time'], axis=1)
    
    # Load correct scaler
    scaler = joblib.load(f"model/scaler_{bank}.pkl")
    row_scaled = scaler.transform(row_features)
    
    return torch.FloatTensor(row_scaled), row

# ── UI: SELECT BANK ─────────────────────────────────
bank = st.selectbox(
    "Select Bank",
    options=["a", "b", "c"],
    format_func=lambda x: {"a": "Bank A (SBI)", "b": "Bank B (HDFC)", "c": "Bank C (ICICI)"}[x]
)

# ── GENERATE TRANSACTION ────────────────────────────
if st.button("🎲 Generate Random Transaction"):
    
    input_tensor, raw_row = load_random_transaction(bank)
    
    st.subheader("📊 Selected Transaction (Raw Data)")
    st.dataframe(raw_row)
    
    # ── PREDICTION FUNCTION ─────────────────────
    def predict(model, x):
      with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()   # ✅ THIS LINE
        prob = min(max(prob, 0.001), 0.999)     # ✅ optional smoothing
        label = "🔴 Fraud" if prob > 0.2 else "🟢 Legit"
      return prob, label
    
    # Get predictions
    results = {
        "Bank A Local": predict(model_a, input_tensor),
        "Bank B Local": predict(model_b, input_tensor),
        "Bank C Local": predict(model_c, input_tensor),
        "🌍 Global Model": predict(global_model, input_tensor),
    }
    
    # ── DISPLAY RESULTS ─────────────────────────
    st.subheader("🔍 Model Comparison Results")
    
    cols = st.columns(4)

    
    for col, (name, (prob, label)) in zip(cols, results.items()):
        with col:
            st.markdown(f"### {name}")
            st.metric("Prediction", label)
            st.metric("Fraud Probability", f"{prob:.4f}")