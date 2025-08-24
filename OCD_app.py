import streamlit as st
import pandas as pd
import joblib
import time

# Load the trained model
model = joblib.load("best_svc_model.joblib")

# Streamlit page config
st.set_page_config(page_title="OCD Patient Medication Prediction", page_icon="🧠", layout="centered")
st.title("🧠 OCD Patient Medication Prediction")
st.markdown("**Machine Learning Model: Support Vector Classifier (SVC)**")
st.info("Enter patient details in the sidebar to predict the suitable medication category.")

# Sidebar for inputs
with st.sidebar:
    st.header("📊 Patient Information")

    # Replace these input fields with the exact features from your dataset
    age = st.slider("Age", 10, 80, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    duration = st.slider("Duration of Illness (months)", 1, 120, 12)
    severity = st.slider("Severity Score", 0, 100, 50)
    family_history = st.selectbox("Family History of OCD", ["Yes", "No"])
    prev_treatment = st.selectbox("Previous Treatment", ["Yes", "No"])

# Prepare input for model
def preprocess_inputs(age, gender, duration, severity, family_history, prev_treatment):
    # Encode categorical variables like in your notebook
    gender_encoded = 1 if gender == "Male" else 0
    family_history_encoded = 1 if family_history == "Yes" else 0
    prev_treatment_encoded = 1 if prev_treatment == "Yes" else 0
    
    return pd.DataFrame([[age, gender_encoded, duration, severity, family_history_encoded, prev_treatment_encoded]],
                        columns=["Age", "Gender", "Duration", "Severity", "Family_History", "Previous_Treatment"])

# Predict button
if st.button("🚀 Predict"):
    input_df = preprocess_inputs(age, gender, duration, severity, family_history, prev_treatment)

    # Simulate a loading animation
    with st.spinner("Calculating prediction..."):
        time.sleep(1)
        prediction = model.predict(input_df)[0]

    # Stylish output display
    st.success("✅ Prediction Completed!")
    st.markdown(
        f"""
        <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:#1E90FF;">Predicted Medication</h2>
            <h1 style="color:#FF4500;font-size:50px;">{prediction}</h1>
            <p style="color:gray;">This is the suggested medication category for the patient</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Optional metric view
    st.metric(label="💊 Predicted Medication", value=prediction)

