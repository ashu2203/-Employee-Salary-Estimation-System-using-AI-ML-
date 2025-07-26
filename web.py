import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from sklearn.preprocessing import LabelEncoder # Explicitly import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Employee Salary Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Load Model and Preprocessing Components ---
model = None
label_encoders = {}
scaler = None
feature_names = []

try:
    model_data = joblib.load("salary_predictor.pkl")
    model = model_data["model"]
    label_encoders = model_data["label_encoders"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]
except FileNotFoundError:
    st.error("Error: 'salary_predictor.pkl' not found. Please ensure your trained model file is in the same directory.")
    st.stop() # Stop the app if the model isn't found
except KeyError as e:
    st.error(f"Error loading model components from 'salary_predictor.pkl': Missing key {e}. Ensure the PKL file contains 'model', 'label_encoders', 'scaler', and 'feature_names'.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    st.stop()

# --- Header ---
st.title("💼 EMPLOYEE SALARY PREDICTOR") 
st.header("📝 Enter Employee Details") # Using Streamlit's native header

# --- Input Form ---
with st.form("salary_form"):
    age = st.slider("Age", min_value=18, max_value=80, value=30)
    
    # Ensure label_encoders has the key before accessing .classes_
    # Provide fallback options for selectbox if encoders are not perfectly loaded
    gender_options = label_encoders.get("Gender", LabelEncoder()).classes_
    gender = st.selectbox("Gender", options=gender_options if len(gender_options) > 0 else ["Male", "Female"])
    
    education_options = label_encoders.get("Education Level", LabelEncoder()).classes_
    education_level = st.selectbox("Education Level", options=education_options if len(education_options) > 0 else ["High School", "Bachelors", "Masters", "PhD"])
    
    job_title_options = label_encoders.get("Job Title", LabelEncoder()).classes_
    job_title = st.selectbox("Job Title", options=job_title_options if len(job_title_options) > 0 else ["Software Engineer", "Data Scientist", "Project Manager", "HR Specialist"])
    
    years_of_experience = st.slider("Years of Experience", min_value=0, max_value=40, value=5)

    # Streamlit buttons are typically centered by default within a form or can be centered using st.columns
    submit_button = st.form_submit_button("Predict Salary")

if submit_button:
    input_data_dict = {
        "Age": age,
        "Gender": gender,
        "Education Level": education_level,
        "Job Title": job_title,
        "Years of Experience": years_of_experience
    }
    input_df = pd.DataFrame([input_data_dict])  # Define input_df first

    # 🔽 Now it's safe to encode
    for col in ["Gender", "Education Level", "Job Title"]:
        try:
            if col in label_encoders and len(label_encoders[col].classes_) > 0:
                input_df[col] = label_encoders[col].transform(input_df[col])
            else:
                st.error(f"LabelEncoder for '{col}' is not properly loaded or has no classes. Cannot encode.")
                st.stop()
        except ValueError as e:
            st.error(f"Error encoding '{col}': {e}.")
            st.stop()

    # Feature order fix
    if feature_names and len(feature_names) == len(input_df.columns):
        input_df = input_df[feature_names]
    else:
        st.warning("Warning: Feature names from model not available or mismatch. Assuming input column order is correct.")

    # Apply Scaling
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Error during scaling: {e}. Please check your scaler and input data format.")
        st.stop()

    # Make Prediction
    try:
        predicted_salary = model.predict(input_scaled)[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}. Ensure the model is correctly loaded and input data is formatted as expected.")
        st.stop()

    # Conversion to INR 
    usd_to_inr_rate = 86.30  # As of 2025

    try:
        predicted_salary_inr = predicted_salary * usd_to_inr_rate
    except Exception as e:
        st.error(f"Error during currency conversion: {e}. Ensure the exchange rate is a valid number.")
        st.stop()

    # --- Display in Streamlit ---
    st.success(f"""
    💰 **Estimated Annual Salary:**
    USD ${predicted_salary:,.4f}\n
    INR ₹{predicted_salary_inr:,.2f}""")
