

import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# ---------------------------------------------------------------------------
# Load trained model, scaler, and imputation medians.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    medians = pd.read_csv("feature_medians.csv", index_col=0).iloc[:, 0]
    return model, scaler, medians

model, scaler, medians = load_artifacts()

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🩺 Diabetes Risk Predictor")
st.write(
    "This app uses a **Logistic Regression** model trained on the Pima Indians "
    "Diabetes dataset to estimate the probability that a patient has diabetes, "
    "based on diagnostic measurements. Enter the patient's values below."
)

st.divider()
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin (mu U/mL)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)

st.caption(
    "Tip: entering 0 for Glucose, Blood Pressure, Skin Thickness, Insulin or BMI "
    "will be treated as a missing value and imputed with the training median, "
    "matching how the model was trained."
)

st.divider()

if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
    input_df = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], columns=FEATURE_NAMES)

    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    input_df[zero_as_missing] = input_df[zero_as_missing].replace(0, np.nan)
    input_df = input_df.fillna(medians)

    input_scaled = scaler.transform(input_df)

    proba = model.predict_proba(input_scaled)[0, 1]
    prediction = model.predict(input_scaled)[0]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ Higher-risk prediction: **Diabetic** (probability = {proba:.1%})")
    else:
        st.success(f"✅ Lower-risk prediction: **Not Diabetic** (probability = {proba:.1%})")

    st.progress(min(max(proba, 0.0), 1.0))
    st.caption(
        "This prediction is for educational purposes only and is not a medical "
        "diagnosis. Please consult a healthcare professional for medical advice."
    )

st.divider()
with st.expander("About this model"):
    st.write(
        """
        - **Algorithm:** Logistic Regression (scikit-learn)
        - **Preprocessing:** Median imputation for missing/implausible values,
          then standard scaling
        - **Training data:** Pima Indians Diabetes Dataset (768 patients, 8 features)
        - **Evaluation (held-out test set):** ~71% accuracy, ~0.81 ROC-AUC
        """
    )
