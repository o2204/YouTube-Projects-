import streamlit as st
import numpy as np
import joblib

# ====================================
# 🚀 Load Model
# ====================================

model = joblib.load(r"E:\AI\Youtube_Projects\YouTube Projects\Heart_Disease\Heart Disease.pkl")

# ====================================
# 🧠 Streamlit UI
# ====================================

st.title("Heart Disease Prediction App")

# User Inputs
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
sex = 0 if sex == "Female" else 1

chest_pain_type = st.selectbox("Chest Pain Type", [
    "Typical Angina (0)",
    "Atypical Angina (1)",
    "Non-anginal Pain (2)",
    "Asymptomatic (3)"
])
chest_pain_type = int(chest_pain_type[-2])  # Extract number from label

resting_blood_pressure = st.slider("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.slider("Cholesterol", 100, 400, 200)

fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
fasting_blood_sugar = int(fasting_blood_sugar[-2])

resting_ecg = st.selectbox("Resting Electrocardiogram", [
    "Normal (0)",
    "ST-T Wave Abnormality (1)",
    "Left Ventricular Hypertrophy (2)"
])
resting_ecg = int(resting_ecg[-2])

max_heart_rate = st.slider("Max Heart Rate Achieved", 70, 210, 150)

exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
exercise_induced_angina = int(exercise_induced_angina[-2])

st_depression = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)

st_slope = st.selectbox("ST Slope", [
    "Upsloping (0)",
    "Flat (1)",
    "Downsloping (2)"
])
st_slope = int(st_slope[-2])

num_major_vessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

thalassemia = st.selectbox("Thalassemia", [
    "Normal (0)",
    "Fixed Defect (1)",
    "Reversible Defect (2)"
])
thalassemia = int(thalassemia[-2])

# Combine all features in the correct order
features = np.array([[age, sex, chest_pain_type, resting_blood_pressure, cholesterol,
                      fasting_blood_sugar, resting_ecg, max_heart_rate,
                      exercise_induced_angina, st_depression, st_slope,
                      num_major_vessels, thalassemia]])

# Predict
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("⚠ High risk of heart disease")
    else:
        st.success("✅ Low risk of heart disease")



st.markdown("---")
st.write("Made with ❤ by Omar Atef")

st.markdown(
    """
    <div style="display: flex; gap: 10px; align-items: center;">
        <a href="https://github.com/o2204" target="_blank">
            <img src="https://img.icons8.com/ios-glyphs/30/000000/github.png" alt="GitHub"/>
        </a>
        <a href="https://www.kaggle.com/omaratef200" target="_blank">
            <img src="https://img.icons8.com/ios-filled/30/000000/linkedin.png" alt="LinkedIn"/>
        </a>
        <a href="https://youtube.com/@omaratef2278?si=i4-m4RY6dK-GAzq6" target="_blank">
            <img src="https://img.icons8.com/ios-filled/30/000000/youtube-play.png" alt="YouTube"/>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
