# CodeAlpha_Project-Disease-Prediction-from-Medical-DataTask3.
Excited to share my latest Machine Learning project. Thank you @CodeAlpha for this fantastic assignment to work on practical, real-world data science challenges. CodeAlpha_Project  Disease Prediction from Medical DataTask3.
Author- Prem Somnath Sonawane



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# --- MODEL TRAINING AND LOADING FUNCTION ---
@st.cache_resource
def load_or_train_model():
    model_path = 'diabetes_model.pkl'
    scaler_path = 'scaler.pkl'
    
    # If the model files don't exist, train them automatically right now
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        # Load dataset
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age', 'Outcome']
        df = pd.read_csv(url, names=columns)
        
        X = df.drop(columns=['Outcome'])
        y = df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save them
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
    # Load and return the working assets
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    return loaded_model, loaded_scaler

# Load the assets safely
model, scaler = load_or_train_model()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Disease Prediction App", layout="centered")
st.title("🩺 Medical Disease Prediction Application")
st.markdown("Input the patient's clinical data below to predict the likelihood of Diabetes.")
st.write("---")

st.subheader("Patient Clinical Metrics")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)

st.write("---")

if st.button("Predict Results", type="primary"):
    # Create feature array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
    
    # Scale input using the safely loaded scaler
    scaled_input = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk Detected.** The model predicts a high probability of Diabetes.")
        st.write(f"**Confidence Level:** {prediction_proba[0][1] * 100:.2f}% risk probability.")
    else:
        st.success(f"✅ **Low Risk Detected.** The model predicts the patient is unlikely to have Diabetes.")
        st.write(f"**Confidence Level:** {prediction_proba[0][0] * 100:.2f}% healthy probability.")
        
        
        #command to run the app:
        #cd disease_prediction
        # streamlit run disease_prediction.py
