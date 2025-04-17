import streamlit as st
import pandas as pd
import base64
import boto3
import io

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import DiabetesData
from src.entity.s3_estimator import Proj1Estimator
from src.entity.config_entity import DiabetesPredictorConfig

# Page setup
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Load style.css
def load_css():
    with open("static/css/style1.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Show banner
def show_banner():
    with open("static/images/banner.jpg", "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
        st.markdown(
            f'<div class="banner"><img src="data:image/jpg;base64,{encoded}" class="banner-image"></div>',
            unsafe_allow_html=True,
        )

# Show logo
def show_logo():
    with open("static/images/logo.png", "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
        st.markdown(
            f'<div class="logo"><img src="data:image/png;base64,{encoded}" class="logo-image"></div>',
            unsafe_allow_html=True,
        )

# Load model from S3
@st.cache_resource
def load_model():
    config = DiabetesPredictorConfig()

    aws_access_key = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
    aws_secret_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    response = s3.get_object(Bucket=config.model_bucket_name, Key=config.model_file_path)
    model_bytes = io.BytesIO(response["Body"].read())

    return Proj1Estimator(model_bytes)

# Main UI
def main():
    load_css()
    show_banner()
    show_logo()

    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown("<h2>Diabetes Prediction Form</h2>", unsafe_allow_html=True)

    with st.form("diabetes_form"):
        st.number_input("Pregnancies", key="Pregnancies", min_value=0, step=1)
        st.number_input("Blood Pressure", key="BloodPressure", min_value=0)
        st.number_input("Skin Thickness", key="SkinThickness", min_value=0)
        st.number_input("Diabetes Pedigree Function", key="DiabetesPedigreeFunction", min_value=0.0, step=0.01)
        st.number_input("Age", key="Age", min_value=1)

        col1, col2 = st.columns(2)
        with col1:
            st.number_input("BMI (optional)", key="BMI", min_value=10.0, max_value=50.0, step=0.1)
        with col2:
            st.selectbox("BMI Category", options=["Underweight", "Normal", "Overweight", "Obesity_type1", "Obesity_type2"], key="NewBMI")

        col3, col4 = st.columns(2)
        with col3:
            st.number_input("Insulin (optional)", key="Insulin", min_value=0, max_value=500)
        with col4:
            st.selectbox("Insulin Category", options=["Normal", "Abnormal"], key="NewInsulinScore")

        col5, col6 = st.columns(2)
        with col5:
            st.number_input("Glucose (optional)", key="Glucose", min_value=0, max_value=300)
        with col6:
            st.selectbox("Glucose Category", options=["Low", "Normal", "Overweight", "Secret", "High"], key="NewGlucose")

        submitted = st.form_submit_button("Predict")

    if submitted:
        data = {
            "Pregnancies": st.session_state["Pregnancies"],
            "BloodPressure": st.session_state["BloodPressure"],
            "SkinThickness": st.session_state["SkinThickness"],
            "DiabetesPedigreeFunction": st.session_state["DiabetesPedigreeFunction"],
            "Age": st.session_state["Age"],
            "BMI": st.session_state["BMI"],
            "NewBMI": st.session_state["NewBMI"],
            "Insulin": st.session_state["Insulin"],
            "NewInsulinScore": st.session_state["NewInsulinScore"],
            "Glucose": st.session_state["Glucose"],
            "NewGlucose": st.session_state["NewGlucose"]
        }

        try:
            diabetes_input = DiabetesData(**data)
            diabetes_df = diabetes_input.get_diabetes_input_data_frame()

            model = load_model()
            value = model.predict(dataframe=diabetes_df)[0]

            if value == 1:
                st.error("⚠️ Patient is **Diabetic**. Please consult your doctor.")
            else:
                st.success("✅ Patient is **Non-Diabetic**. Keep up the healthy lifestyle!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()