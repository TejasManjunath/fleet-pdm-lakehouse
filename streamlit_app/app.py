import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="Fleet PdM — Failure Risk Predictor", layout="centered")

st.title("🚛 Fleet Predictive Maintenance")
st.caption("Cost-optimized APS failure prediction — SCANIA truck fleet, Random Forest model")

# Load sample truck data shipped with the app
@st.cache_data
def load_samples():
    return pd.read_csv("sample_trucks.csv")

samples = load_samples()

st.subheader("Select a truck to evaluate")
idx = st.selectbox(
    "Truck (from held-out test set)",
    options=samples.index,
    format_func=lambda i: f"Truck #{i} — actual label: {'FAILURE' if samples.loc[i, 'label'] == 1 else 'healthy'}"
)

truck_row = samples.loc[[idx]].drop(columns=["label"])
actual_label = samples.loc[idx, "label"]

with st.expander("View raw sensor data for this truck"):
    st.dataframe(truck_row.T, use_container_width=True)

DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]
ENDPOINT_URL = "https://dbc-807047c6-74fb.cloud.databricks.com/serving-endpoints/fleet-pdm-rf-endpoint/invocations"

def score_model(dataset):
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}", "Content-Type": "application/json"}
    ds_dict = {"dataframe_split": dataset.to_dict(orient="split")}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.post(ENDPOINT_URL, headers=headers, data=data_json, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} — {response.text}")
    return response.json()

if st.button("Predict failure risk", type="primary"):
    with st.spinner("Calling live model endpoint (may take up to 30s on cold start)..."):
        try:
            result = score_model(truck_row)
            prediction = result["predictions"][0]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model prediction", "⚠️ FAILURE RISK" if prediction == 1 else "✅ Healthy")
            with col2:
                st.metric("Actual outcome", "FAILURE" if actual_label == 1 else "Healthy")

            if prediction == 1:
                st.warning("Recommendation: schedule preventive APS inspection (~€10 cost) rather than risk a €500 breakdown.")
            else:
                st.success("No action needed — model confidence supports skipping inspection.")

            st.divider()
            st.caption(
                "This model was selected after benchmarking Logistic Regression, Random Forest, and XGBoost "
                "with cost-sensitive threshold optimization against Scania's published cost matrix "
                "(€500 missed failure / €10 unnecessary inspection). On a held-out test set of 16,000 trucks, "
                "this model achieves €637.50 cost per 1,000 trucks — a 93.5% reduction vs. inspecting every truck."
            )
        except Exception as e:
            st.error(f"Error calling model endpoint: {e}")