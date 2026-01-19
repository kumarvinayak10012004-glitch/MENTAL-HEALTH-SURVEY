import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from src.model import MLP

# Paths to artifacts
ARTIFACTS_DIR = "artifacts"
SCHEMA_PATH = f"{ARTIFACTS_DIR}/schema.json"
PREPROCESSOR_PATH = f"{ARTIFACTS_DIR}/preprocessor.joblib"
MODEL_PATH = f"{ARTIFACTS_DIR}/model.pth"

@st.cache_resource
def load_artifacts():
    """Load schema, preprocessor, and model from artifacts folder."""
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    pre = joblib.load(PREPROCESSOR_PATH)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = MLP(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return schema, pre, model

def ui_inputs(schema: dict):
    """Render sidebar input widgets dynamically based on schema."""
    st.sidebar.header("Enter survey features")
    cat_cols = schema["categorical_cols"]
    num_cols = schema["numeric_cols"]

    inputs = {}

    for col in num_cols:
        inputs[col] = st.sidebar.number_input(
            col, value=0.0, step=1.0, format="%.2f"
        )

    for col in cat_cols:
        # free text input is robust to unseen categories
        inputs[col] = st.sidebar.text_input(col, value="")

    return inputs

def predict_row(inputs: dict, schema, pre, model):
    """Predict probability of depression for a single input row."""
    # Build single-row DataFrame
    cols = schema["feature_cols"]
    row = {c: inputs.get(c, "") for c in cols}
    df = pd.DataFrame([row], columns=cols)

    # Preprocess
    X = pre.transform(df)

    # Model inference
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        prob = torch.sigmoid(logits).item()
    return prob

def risk_category(prob, thresholds=(0.33, 0.66)):
    """Classify probability into risk categories."""
    if prob < thresholds[0]:
        return "Low"
    elif prob < thresholds[1]:
        return "Medium"
    else:
        return "High"

def main():
    st.title("🧠 Depression Risk Predictor")
    st.write("Estimate the likelihood of depression based on survey responses.")

    try:
        schema, pre, model = load_artifacts()
    except Exception as e:
        st.error(f"Artifacts not found: {e}. Please train the model first.")
        st.stop()

    # Inputs
    inputs = ui_inputs(schema)

    # Optional: adjustable threshold
    threshold = st.sidebar.slider("Positive Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

    if st.button("Predict"):
        prob = predict_row(inputs, schema, pre, model)
        st.metric("Predicted Probability of Depression", f"{prob:.2%}")
        st.write("Prediction:", "Positive" if prob >= threshold else "Negative")
        st.write("Risk Category:", risk_category(prob))

if __name__ == "__main__":
    main()
