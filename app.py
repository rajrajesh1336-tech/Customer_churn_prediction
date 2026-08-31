import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# The model is expected to be kept beside this app.
MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"

# Exact feature schema used by Model_Training.ipynb after preprocessing:
# SeniorCitizen is converted from int64 (0/1) to categorical Yes/No before training.
FEATURE_DTYPES = {
    "gender": "object",
    "SeniorCitizen": "object",
    "Partner": "object",
    "Dependents": "object",
    "tenure": "int64",
    "PhoneService": "object",
    "MultipleLines": "object",
    "InternetService": "object",
    "OnlineSecurity": "object",
    "OnlineBackup": "object",
    "DeviceProtection": "object",
    "TechSupport": "object",
    "StreamingTV": "object",
    "StreamingMovies": "object",
    "Contract": "object",
    "PaperlessBilling": "object",
    "PaymentMethod": "object",
    "MonthlyCharges": "float64",
    "TotalCharges": "float64",
}

CATEGORICAL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

NUMERIC_INT_COLUMNS = ["tenure"]
NUMERIC_FLOAT_COLUMNS = ["MonthlyCharges", "TotalCharges"]

# Values below are taken from cleaned_data.csv.
DATA_PATH = Path(__file__).resolve().parent / "cleaned_data.csv"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run Model_Training.ipynb first and place best_model.pkl beside app.py."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_values():
    if not DATA_PATH.exists():
        return {}
    data = pd.read_csv(DATA_PATH)
    values = {}
    for col in CATEGORICAL_COLUMNS:
        if col in data.columns:
            if col == "SeniorCitizen":
                values[col] = ["No", "Yes"]
            else:
                values[col] = sorted(data[col].dropna().astype(str).unique().tolist())
    return values


def validate_input(df):
    expected = list(FEATURE_DTYPES.keys())
    actual = list(df.columns)

    if actual != expected:
        missing = [c for c in expected if c not in actual]
        extra = [c for c in actual if c not in expected]
        raise ValueError(f"Feature mismatch. Missing: {missing}; Extra: {extra}")

    # Check the semantic dtypes expected by the training notebook.
    for col in CATEGORICAL_COLUMNS:
        if not pd.api.types.is_object_dtype(df[col]):
            raise TypeError(f"{col} must be categorical/object data.")

    if not pd.api.types.is_integer_dtype(df["tenure"]):
        raise TypeError("tenure must be integer data.")

    for col in NUMERIC_FLOAT_COLUMNS:
        if not pd.api.types.is_float_dtype(df[col]):
            raise TypeError(f"{col} must be float data.")


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict the probability of churn.")

try:
    model = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

feature_values = load_feature_values()

with st.form("customer_form"):
    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", feature_values.get("gender", ["Female", "Male"]))
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", feature_values.get("Partner", ["No", "Yes"]))
        dependents = st.selectbox("Dependents", feature_values.get("Dependents", ["No", "Yes"]))
        tenure = st.number_input("Tenure (months)", min_value=0, step=1, value=1)
        phone_service = st.selectbox("Phone Service", feature_values.get("PhoneService", ["No", "Yes"]))

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            feature_values.get("MultipleLines", ["No", "No phone service", "Yes"]),
        )
        internet_service = st.selectbox(
            "Internet Service",
            feature_values.get("InternetService", ["DSL", "Fiber optic", "No"]),
        )
        online_security = st.selectbox(
            "Online Security",
            feature_values.get("OnlineSecurity", ["No", "No internet service", "Yes"]),
        )
        online_backup = st.selectbox(
            "Online Backup",
            feature_values.get("OnlineBackup", ["No", "No internet service", "Yes"]),
        )
        device_protection = st.selectbox(
            "Device Protection",
            feature_values.get("DeviceProtection", ["No", "No internet service", "Yes"]),
        )
        tech_support = st.selectbox(
            "Tech Support",
            feature_values.get("TechSupport", ["No", "No internet service", "Yes"]),
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            feature_values.get("StreamingTV", ["No", "No internet service", "Yes"]),
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            feature_values.get("StreamingMovies", ["No", "No internet service", "Yes"]),
        )
        contract = st.selectbox(
            "Contract",
            feature_values.get("Contract", ["Month-to-month", "One year", "Two year"]),
        )
        paperless_billing = st.selectbox(
            "Paperless Billing",
            feature_values.get("PaperlessBilling", ["No", "Yes"]),
        )
        payment_method = st.selectbox(
            "Payment Method",
            feature_values.get(
                "PaymentMethod",
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ],
            ),
        )
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, step=0.01, value=50.00, format="%.2f"
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, step=0.01, value=50.00, format="%.2f"
        )

    submitted = st.form_submit_button("Predict Churn", type="primary")

if submitted:
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }])

    # Explicitly enforce the same feature order and data types used during training.
    input_data = input_data[list(FEATURE_DTYPES.keys())]
    input_data["tenure"] = input_data["tenure"].astype("int64")
    for col in CATEGORICAL_COLUMNS:
        input_data[col] = input_data[col].astype("object")
    for col in NUMERIC_FLOAT_COLUMNS:
        input_data[col] = input_data[col].astype("float64")

    try:
        validate_input(input_data)

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0, 1])

        st.subheader("Prediction")

        if prediction == 1:
            st.error("⚠️ Customer is predicted to CHURN")
        else:
            st.success("✅ Customer is predicted to NOT CHURN")

        st.metric("Churn Probability", f"{probability:.2%}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
