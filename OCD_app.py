import streamlit as st
import pandas as pd
import joblib
import time

# Load trained model
model = joblib.load("best_svc_model.joblib")

# Streamlit page settings
st.set_page_config(page_title="OCD Medication Prediction", page_icon="🧠", layout="centered")
st.title("🧠 OCD Patient Medication Prediction")
st.markdown("**Machine Learning Model: Support Vector Classifier (SVC)**")
st.info("Enter patient details in the sidebar to predict the suitable medication.")

# Sidebar inputs
with st.sidebar:
    st.header("📊 Patient Information")

    # Numeric inputs
    age = st.slider("Age", 5, 90, 30)
    ybocs_obs = st.slider("Y-BOCS Score (Obsessions)", 0, 40, 15)
    ybocs_comp = st.slider("Y-BOCS Score (Compulsions)", 0, 40, 15)
    duration = st.slider("Duration of Symptoms (months)", 1, 240, 12)

    # Categorical inputs (placeholders – replace with actual dataset values later)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    ethnicity = st.selectbox("Ethnicity", ["Option1", "Option2", "Option3"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    education = st.selectbox("Education Level", ["High School", "Graduate", "Post-Graduate", "Other"])
    prev_diag = st.selectbox("Previous Diagnoses", ["Yes", "No"])
    family_history = st.selectbox("Family History of OCD", ["Yes", "No"])
    obsession_type = st.selectbox("Obsession Type", ["Option1", "Option2", "Option3"])
    compulsion_type = st.selectbox("Compulsion Type", ["Option1", "Option2", "Option3"])
    depression = st.selectbox("Depression Diagnosis", ["Yes", "No"])
    anxiety = st.selectbox("Anxiety Diagnosis", ["Yes", "No"])

# Convert inputs into a dataframe
def preprocess_input():
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
        "Family History of OCD": [family_history],
        "Obsession Type": [obsession_type],
        "Compulsion Type": [compulsion_type],
        "Depression Diagnosis": [depression],
        "Anxiety Diagnosis": [anxiety]
    }
    return pd.DataFrame(data)

# Prediction button
if st.button("🚀 Predict"):
    input_df = preprocess_input()

    # Apply same encoding as training (get_dummies)
    input_encoded = pd.get_dummies(input_df)

    # Align with model training columns (fill missing)
    model_features = model.feature_names_in_
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    # Loading animation
    with st.spinner("Calculating prediction..."):
        time.sleep(1)
        prediction = model.predict(input_encoded)[0]

    # Stylish output
    st.success("✅ Prediction Completed!")
    st.markdown(
        f"""
        <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:#1E90FF;">Predicted Medication</h2>
            <h1 style="color:#FF4500;font-size:50px;">{prediction}</h1>
            <p style="color:gray;">This is the suggested medication category for the patient</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric(label="💊 Predicted Medication", value=prediction)


