import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

# ==========================================
# 🎯 LOAD MODEL + SCALER + ENCODER
# ==========================================
MODEL_PATH = r"C:\Global_AI_Career_Intelligence\models\gradient_boost_model.pkl"
SCALER_PATH = r"C:\Global_AI_Career_Intelligence\models\scaler.pkl"
ENCODER_PATH = r"C:\Global_AI_Career_Intelligence\models\label_encoder.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODER_PATH)

# ==========================================
# ⚙️ APP CONFIG
# ==========================================
st.set_page_config(page_title="AI Career Intelligence", page_icon="🤖", layout="wide")

# ==========================================
# HEADER
# ==========================================
col1, col2 = st.columns([1, 6])
with col1:
    try:
        img = Image.open(r"C:\Global_AI_Career_Intelligence\PowerBI\ai.png")
        st.image(img, width=400)
    except Exception:
        st.warning("⚠️ Logo not found — skipping image")

with col2:
    st.title("💼 AI Career Intelligence & Salary Prediction")
    st.markdown("### WELCOME TO AI WORLD")

st.markdown("---")

# ==========================================
# SIDEBAR INPUTS
# ==========================================
st.sidebar.header("🔍 Enter Job Details")

job_title = st.sidebar.selectbox("🧠 Job Title", [
    "Data Scientist", "Machine Learning Engineer", "AI Researcher",
    "Data Analyst", "Deep Learning Engineer", "AI Product Manager", "Business Analyst"
])
experience = st.sidebar.slider("💼 Years of Experience", 0, 20, 3)
experience_level = st.sidebar.selectbox("📊 Experience Level", ["EN", "MI", "SE", "EX"])
employment_type = st.sidebar.selectbox("🧾 Employment Type", ["FT", "PT", "CT", "FL"])
education_required = st.sidebar.selectbox("🎓 Education Level", ["Bachelor's", "Master's", "PhD", "Other"])
company_size = st.sidebar.selectbox("🏢 Company Size", ["S", "M", "L"])
company_location = st.sidebar.text_input("📍 Company Location (e.g., US)", "US")
employee_residence = st.sidebar.text_input("🏠 Employee Residence (e.g., IN)", "IN")
remote_ratio = st.sidebar.slider("🌐 Remote Ratio (%)", 0, 100, 50)
industry = st.sidebar.selectbox("🏭 Industry", ["IT", "Business", "Health Care", "Finance", "Education", "Manufacturing"])
job_description_length = st.sidebar.number_input("📝 Job Description Length", min_value=0, max_value=1000, value=200)
benefits_score = st.sidebar.slider("🎁 Benefits Score", 0.0, 1.0, 0.5)
required_skills = st.sidebar.text_area("🧩 Required Skills (comma separated)", "Python, SQL, Power BI, Excel, AI, Deep Learning")

# ==========================================
# DATA PREPARATION
# ==========================================
input_data = pd.DataFrame({
    'job_title': [job_title],
    'salary_usd': [0],
    'salary_currency': ['USD'],
    'experience_level': [experience_level],
    'employment_type': [employment_type],
    'company_location': [company_location],
    'company_size': [company_size],
    'employee_residence': [employee_residence],
    'remote_ratio': [remote_ratio],
    'required_skills': [required_skills],
    'education_required': [education_required],
    'years_experience': [experience],
    'industry': [industry],
    'job_description_length': [job_description_length],
    'benefits_score': [benefits_score],
    'predicted_salary': [0]
})

# ==========================================
# SAFE LABEL ENCODING
# ==========================================
for col in input_data.columns:
    if col in label_encoders:
        try:
            input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
        except Exception:
            input_data[col] = 0  # unseen category fallback

# ==========================================
# ALIGN FEATURES WITH MODEL
# ==========================================
expected_features = getattr(model, "feature_names_in_", input_data.columns)
for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[expected_features]

# ==========================================
# APPLY SCALER SAFELY
# ==========================================
try:
    if hasattr(scaler, "feature_names_in_"):
        for col in scaler.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[scaler.feature_names_in_]
        input_data = pd.DataFrame(
            scaler.transform(input_data),
            columns=scaler.feature_names_in_
        )
except Exception:
    pass  # Skip warning messages

# ==========================================
# PREDICTION
# ==========================================
if st.sidebar.button("🚀 Predict Salary"):
    try:
        predicted_salary = model.predict(input_data)[0]
        st.success(f"💰 **Predicted Annual Salary:** ${predicted_salary:,.2f}")
        st.info("This prediction is based on machine learning model insights — actual values may vary.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ==========================================
# 📊 DASHBOARD IMAGE (instead of chart)
# ==========================================
st.markdown("---")
st.subheader("📊 AI Career Intelligence Dashboard Snapshot")

try:
    dashboard_img = Image.open(r"C:\Global_AI_Career_Intelligence\PowerBI\ai.png")
    st.image(dashboard_img, width='stretch', caption="Global AI Career Insights — Power BI Dashboard")
except Exception:
    st.warning("⚠️ Dashboard image not found. Please check the path.")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("🔧 Developed using Streamlit | Machine Learning Model by Hari Prasad")
