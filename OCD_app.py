import streamlit as st
import pandas as pd
import joblib
import time

# Load trained model
model = joblib.load("best_svc_model.joblib")

# Manual mapping (LabelEncoder mapping)
class_mapping = {0: "Benzodiazepine", 1: "SSRI", 2: "SNRI"}

# Page config
st.set_page_config(page_title="OCD Patient Medication Prediction", page_icon="🧠", layout="centered")
st.title("🧠 OCD Patient Medication Prediction")
st.markdown("**Machine Learning Model: Support Vector Classifier (SVC)**")
st.info("Enter patient details in the sidebar to predict the suitable medication category.")

# Sidebar inputs
with st.sidebar:
    st.header("📊 Patient Information")

    # Numeric inputs
    age = st.slider("Age", 10, 80, 30)
    ybocs_obs = st.slider("Y-BOCS Score (Obsessions)", 0, 40, 10)
    ybocs_comp = st.slider("Y-BOCS Score (Compulsions)", 0, 40, 10)
    duration = st.slider("Duration of Symptoms (months)", 1, 120, 12)

    # Categorical inputs (placeholders – replace with dataset values if needed)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", ["African", "Asian", "Hispanic", "Caucasian"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = st.selectbox("Education Level", ["Some College", "College Degree", "High School", "Graduate Degree"])
    prev_diag = st.selectbox("Previous Diagnoses", ["Yes", "No"])
    fam_history = st.selectbox("Family History of OCD", ["Yes", "No"])
    obsession = st.selectbox("Obsession Type", ["Harm-related", "Contamination", "Symmetry", "Hoarding"], "Religious")
    compulsion = st.selectbox("Compulsion Type", ["Checking", "Washing", "Ordering", "Praying", "Counting"])
    depression = st.selectbox("Depression Diagnosis", ["Yes", "No"])
    anxiety = st.selectbox("Anxiety Diagnosis", ["Yes", "No"])

# Preprocessing: Convert inputs to DataFrame
def preprocess_inputs():
    data = {
        "Age": [age],
        "Y-BOCS Score (Obsessions)": [ybocs_obs],
        "Y-BOCS Score (Compulsions)": [ybocs_comp],
        "Duration of Symptoms (months)": [duration],
        "Gender": [gender],
        "Ethnicity": [ethnicity],
        "Marital Status": [marital_status],
        "Education Level": [education],
        "Previous Diagnoses": [prev_diag],
        "Family History of OCD": [fam_history],
        "Obsession Type": [obsession],
        "Compulsion Type": [compulsion],
        "Depression Diagnosis": [depression],
        "Anxiety Diagnosis": [anxiety],
    }
    return pd.DataFrame(data)

# Prediction button
if st.button("🚀 Predict"):
    input_df = preprocess_inputs()

    with st.spinner("Calculating prediction..."):
        time.sleep(1)
        prediction_num = model.predict(input_df)[0]

        # Map prediction number to medication class
        prediction_label = class_mapping.get(prediction_num, f"Class {prediction_num}")

    st.success("✅ Prediction Completed!")
    st.markdown(
        f"""
        <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:#1E90FF;">Predicted Medication</h2>
            <h1 style="color:#FF4500;font-size:50px;">{prediction_label}</h1>
            <p style="color:gray;">This is the suggested medication category for the patient</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric(label="💊 Predicted Medication", value=prediction_label)



