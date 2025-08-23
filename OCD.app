import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="OCD Patient Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; font-weight: bold;}
    .section-header {font-size: 1.8rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 0.3rem;}
    .info-text {font-size: 1.1rem; line-height: 1.6;}
    .prediction-box {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;}
    .positive {color: #ff4b4b; font-weight: bold;}
    .negative {color: #0068c9; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">🧠 OCD Patient Prediction Using Machine Learning</p>', unsafe_allow_html=True)
st.markdown('<p class="info-text">This application predicts the likelihood of OCD diagnosis based on patient data using a trained machine learning model.</p>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:", 
                           ["Home", "Data Exploration", "Make Prediction", "About"])

# Load data and model
@st.cache_data
def load_data():
    data = pd.read_csv('OCD Patient Dataset.csv')
    return data

@st.cache_resource
def load_model():
    try:
        with open('ocd_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except:
        st.error("Model file not found. Please ensure 'ocd_model.pkl' is in the same directory.")
        return None

# Load data and model
df = load_data()
model = load_model()

if options == "Home":
    st.markdown('<p class="section-header">Dataset Overview</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Number of Features:**", len(df.columns))
        st.write("**First 5 Rows:**")
        st.dataframe(df.head(), height=200)
    
    with col2:
        st.write("**Dataset Information:**")
        # Create a string buffer to capture info
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    
    st.markdown('<p class="section-header">Missing Values</p>', unsafe_allow_html=True)
    missing_data = df.isnull().sum()
    st.bar_chart(missing_data)

elif options == "Data Exploration":
    st.markdown('<p class="section-header">Data Exploration</p>', unsafe_allow_html=True)
    
    # Age Distribution
    st.subheader("Age Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Age'], bins=20, kde=True, ax=ax)
    ax.set_title('Age Distribution')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    # Gender Distribution
    st.subheader("Gender Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Gender', data=df, palette=['red', 'green'], ax=ax)
    ax.set_title('Gender Distribution')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    # Ethnicity Distribution
    st.subheader("Ethnicity Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Ethnicity', data=df, palette=['yellow', 'red', 'blue', 'green'], ax=ax)
    ax.set_title('Ethnicity Distribution')
    ax.set_xlabel('Ethnicity')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Y-BOCS Scores Distribution
    st.subheader("Y-BOCS Scores Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Y-BOCS Score (Obsessions)'], bins=20, kde=True, ax=ax)
        ax.set_title('Y-BOCS Obsessions Score Distribution')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Y-BOCS Score (Compulsions)'], bins=20, kde=True, ax=ax)
        ax.set_title('Y-BOCS Compulsions Score Distribution')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

elif options == "Make Prediction":
    st.markdown('<p class="section-header">Make a Prediction</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please ensure the model file is available.")
    else:
        # Create input form
        with st.form("prediction_form"):
            st.subheader("Patient Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", min_value=18, max_value=75, value=32)
                gender = st.selectbox("Gender", options=["Male", "Female"])
                ethnicity = st.selectbox("Ethnicity", options=["African", "Hispanic", "Caucasian", "Asian"])
                marital_status = st.selectbox("Marital Status", options=["Single", "Divorced", "Married"])
                education_level = st.selectbox("Education Level", options=["Some College", "College Degree", "Graduate Degree", "High School"])
            
            with col2:
                duration_symptoms = st.slider("Duration of Symptoms (months)", min_value=6, max_value=240, value=180)
                family_history = st.selectbox("Family History of OCD", options=["No", "Yes"])
                obsession_type = st.selectbox("Obsession Type", options=["Harm-related", "Contamination", "Symmetry", "Hoarding", "Religious"])
                compulsion_type = st.selectbox("Compulsion Type", options=["Checking", "Washing", "Counting", "Ordering", "Repeating"])
                depression_diagnosis = st.selectbox("Depression Diagnosis", options=["No", "Yes"])
                anxiety_diagnosis = st.selectbox("Anxiety Diagnosis", options=["No", "Yes"])
            
            st.subheader("Y-BOCS Scores")
            col3, col4 = st.columns(2)
            
            with col3:
                obsession_score = st.slider("Y-BOCS Obsession Score", min_value=0, max_value=40, value=10)
            
            with col4:
                compulsion_score = st.sllider("Y-BOCS Compulsion Score", min_value=0, max_value=40, value=10)
            
            submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Prepare the input data for prediction
            input_data = {
                'Age': age,
                'Gender': 1 if gender == "Female" else 0,
                'Ethnicity_African': 1 if ethnicity == "African" else 0,
                'Ethnicity_Asian': 1 if ethnicity == "Asian" else 0,
                'Ethnicity_Caucasian': 1 if ethnicity == "Caucasian" else 0,
                'Ethnicity_Hispanic': 1 if ethnicity == "Hispanic" else 0,
                'Marital Status_Divorced': 1 if marital_status == "Divorced" else 0,
                'Marital Status_Married': 1 if marital_status == "Married" else 0,
                'Marital Status_Single': 1 if marital_status == "Single" else 0,
                'Education Level_College Degree': 1 if education_level == "College Degree" else 0,
                'Education Level_Graduate Degree': 1 if education_level == "Graduate Degree" else 0,
                'Education Level_High School': 1 if education_level == "High School" else 0,
                'Education Level_Some College': 1 if education_level == "Some College" else 0,
                'Duration of Symptoms (months)': duration_symptoms,
                'Family History of OCD_No': 1 if family_history == "No" else 0,
                'Family History of OCD_Yes': 1 if family_history == "Yes" else 0,
                'Obsession Type_Contamination': 1 if obsession_type == "Contamination" else 0,
                'Obsession Type_Harm-related': 1 if obsession_type == "Harm-related" else 0,
                'Obsession Type_Hoarding': 1 if obsession_type == "Hoarding" else 0,
                'Obsession Type_Religious': 1 if obsession_type == "Religious" else 0,
                'Obsession Type_Symmetry': 1 if obsession_type == "Symmetry" else 0,
                'Compulsion Type_Checking': 1 if compulsion_type == "Checking" else 0,
                'Compulsion Type_Counting': 1 if compulsion_type == "Counting" else 0,
                'Compulsion Type_Ordering': 1 if compulsion_type == "Ordering" else 0,
                'Compulsion Type_Repeating': 1 if compulsion_type == "Repeating" else 0,
                'Compulsion Type_Washing': 1 if compulsion_type == "Washing" else 0,
                'Depression Diagnosis_No': 1 if depression_diagnosis == "No" else 0,
                'Depression Diagnosis_Yes': 1 if depression_diagnosis == "Yes" else 0,
                'Anxiety Diagnosis_No': 1 if anxiety_diagnosis == "No" else 0,
                'Anxiety Diagnosis_Yes': 1 if anxiety_diagnosis == "Yes" else 0,
                'Y-BOCS Score (Obsessions)': obsession_score,
                'Y-BOCS Score (Compulsions)': compulsion_score
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure all columns are present (fill missing with 0)
            model_columns = model.feature_names_in_
            for col in model_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match model training
            input_df = input_df[model_columns]
            
            # Make prediction
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            # Display results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader("Prediction Results")
            
            if prediction[0] == 1:
                st.markdown(f'<p class="positive">Prediction: High likelihood of OCD diagnosis</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="negative">Prediction: Low likelihood of OCD diagnosis</p>', unsafe_allow_html=True)
            
            st.write(f"Probability of OCD: {prediction_proba[0][1]*100:.2f}%")
            st.write(f"Probability of no OCD: {prediction_proba[0][0]*100:.2f}%")
            
            # Show probability chart
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = ['No OCD', 'OCD']
            probabilities = [prediction_proba[0][0]*100, prediction_proba[0][1]*100]
            colors = ['#0068c9', '#ff4b4b']
            
            bars = ax.bar(labels, probabilities, color=colors)
            ax.set_ylabel('Probability (%)')
            ax.set_title('Prediction Probability')
            
            # Add value labels on bars
            for bar, probability in zip(bars, probabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{probability:.2f}%', ha='center', va='bottom')
            
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

elif options == "About":
    st.markdown('<p class="section-header">About This Application</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-text">
    <p>This application uses a machine learning model to predict the likelihood of Obsessive-Compulsive Disorder (OCD) diagnosis based on patient data.</p>
    
    <p><strong>Features used in the model:</strong></p>
    <ul>
        <li>Demographic information (Age, Gender, Ethnicity)</li>
        <li>Marital status and education level</li>
        <li>Duration of symptoms</li>
        <li>Family history of OCD</li>
        <li>Obsession and compulsion types</li>
        <li>Y-BOCS scores for obsessions and compulsions</li>
        <li>Depression and anxiety diagnoses</li>
    </ul>
    
    <p><strong>Y-BOCS Scale:</strong> The Yale-Brown Obsessive Compulsive Scale is used to measure the severity of OCD symptoms, with scores ranging from 0 to 40.</p>
    
    <p>This tool is intended for educational and research purposes only and should not replace professional medical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("For more information about OCD, please consult a healthcare professional.")

# Footer
st.markdown("---")
st.markdown("© 2023 OCD Prediction App | For Educational Purposes Only")
