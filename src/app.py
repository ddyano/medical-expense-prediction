import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "models" / "gradient_boosting_model.pkl")
preprocessor = joblib.load(BASE_DIR / "models" / "preprocessor.pkl")
st.set_page_config(
    page_title="Medical Expense Prediction",
    page_icon="💰"
)
st.title("Medical Expense Prediction")
st.write("Enter patient details to predict estimated medical expenses.")
print("Streamlit app started!")


age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)

gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.5,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

smoker = st.selectbox(
    "Smoker Status",
    ["non-smoker", "smoker"]
)

if st.button("Predict Medical Expense"):
    patient = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "children": [children],
        "region": [region],
        "smoker": [smoker]
    })

    st.write(patient)

    patient_processed = preprocessor.transform(patient)
    prediction = model.predict(patient_processed)
    st.success(f"Estimated Medical Expense: ₹{prediction[0]:,.2f}")

