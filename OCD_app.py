
import streamlit as st
import pandas as pd
import joblib
import time
from typing import List, Optional

st.set_page_config(page_title="OCD Patient Medication Prediction", page_icon="🧠", layout="centered")
st.title("🧠 OCD Patient Medication Prediction")
st.markdown("**Machine Learning Model: Support Vector Classifier (SVC)**")
st.info("Enter patient details in the sidebar to predict the suitable medication category.")

# ---------- Utilities ----------

NUMERIC_COLS: List[str] = [
    "Age",
    "Y-BOCS Score (Obsessions)",
    "Y-BOCS Score (Compulsions)",
    "Duration of Symptoms (months)",
]

CATEGORICAL_COLS: List[str] = [
    "Gender",
    "Ethnicity",
    "Marital Status",
    "Education Level",
    "Previous Diagnoses",
    "Family History of OCD",
    "Obsession Type",
    "Compulsion Type",
    "Depression Diagnosis",
    "Anxiety Diagnosis",
]

# Sidebar inputs
with st.sidebar:
    st.header("📊 Patient Information")

    # Numeric inputs
    age = st.slider("Age", 10, 80, 30)
    ybocs_obs = st.slider("Y-BOCS Score (Obsessions)", 0, 40, 10)
    ybocs_comp = st.slider("Y-BOCS Score (Compulsions)", 0, 40, 10)
    duration = st.slider("Duration of Symptoms (months)", 1, 120, 12)

    # Categorical inputs (choices approximated from your notebook)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", ["African", "Asian", "Hispanic", "Caucasian"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = st.selectbox("Education Level", ["Some College", "College Degree", "High School", "Graduate Degree"])
    prev_diag = st.selectbox("Previous Diagnoses", ["Yes", "No"])
    fam_history = st.selectbox("Family History of OCD", ["Yes", "No"])
    # FIX: the old code passed a string as the 3rd positional arg (index) and missed 'Religious' in options.
    obsession = st.selectbox(
        "Obsession Type",
        ["Harm-related", "Contamination", "Symmetry", "Hoarding", "Religious"],
        index=0,
    )
    compulsion = st.selectbox("Compulsion Type", ["Checking", "Washing", "Ordering", "Praying", "Counting"])
    depression = st.selectbox("Depression Diagnosis", ["Yes", "No"])
    anxiety = st.selectbox("Anxiety Diagnosis", ["Yes", "No"])

def to_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
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
    )

@st.cache_resource(show_spinner=False)
def load_model(model_path: str = "best_svc_model.joblib"):
    try:
        mdl = joblib.load(model_path)
        return mdl, None
    except Exception as e:
        return None, e

def get_model_features(model) -> Optional[List[str]]:
    # If the model was trained with a pandas DataFrame, scikit-learn stores feature_names_in_
    feats = getattr(model, "feature_names_in_", None)
    if feats is not None:
        return list(feats)
    # Fallback: try a sidecar file with saved columns list
    sidecar_candidates = ["feature_columns.json", "features.json", "columns.json"]
    for p in sidecar_candidates:
        fp = Path(p)
        if fp.exists():
            try:
                return list(pd.read_json(fp, typ="series").tolist())
            except Exception:
                try:
                    return list(pd.read_json(fp).iloc[:, 0].tolist())
                except Exception:
                    pass
    return None

def preprocess(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    # One-hot encode the categorical columns exactly like notebook (pd.get_dummies)
    encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    # Align to training columns
    encoded_aligned = encoded.reindex(columns=expected_cols, fill_value=0)
    # Ensure numeric dtypes
    encoded_aligned[NUMERIC_COLS] = encoded_aligned[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
    return encoded_aligned

# Load model once
model, load_err = load_model()
if load_err is not None:
    st.error(f"❌ Could not load model file 'best_svc_model.joblib'. Details: {load_err}")
    st.stop()

# Try to get feature names used during training
feature_cols = get_model_features(model)
if feature_cols is None:
    st.error(
        "❌ Unable to determine the model's training feature names. "
        "Please retrain saving feature names or ensure the model was trained with a pandas DataFrame "
        "(so .feature_names_in_ is available)."
    )
    st.stop()

# Mapping from numeric classes to medication labels.
# NOTE: This assumes the label encoding order from training. Adjust if your notebook used a different order.
CLASS_MAPPING = {
    0: "Benzodiazepine",
    1: "SSRI",
    2: "SNRI",
}

# Prediction button
if st.button("🚀 Predict"):
    input_df = to_dataframe()
    with st.spinner("Calculating prediction..."):
        time.sleep(0.6)
        try:
            X = preprocess(input_df, feature_cols)
            pred_num = int(model.predict(X)[0])
            pred_label = CLASS_MAPPING.get(pred_num, f"Class {pred_num}")
            st.success("✅ Prediction Completed!")
            st.markdown(
                f"""
                <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;text-align:center;">
                    <h2 style="color:#1E90FF;">Predicted Medication</h2>
                    <h1 style="color:#FF4500;font-size:50px;">{pred_label}</h1>
                    <p style="color:gray;">This is the suggested medication category for the patient.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric(label="💊 Predicted Medication", value=pred_label)
        except Exception as e:
            st.error(
                "Prediction failed. This usually means your input columns don't match the model's training columns.\n\n"
                f"**Details:** {e}"
            )
            with st.expander("Show prepared input features"):
                st.write(preprocess(input_df, feature_cols).head())
