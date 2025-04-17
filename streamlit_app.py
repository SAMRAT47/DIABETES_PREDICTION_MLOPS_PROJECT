import streamlit as st
import base64
import pandas as pd
import sys
import os
import boto3
from io import BytesIO
import pickle

# Custom exception class
class MyException(Exception):
    def __init__(self, error_message, error_detail):
        self.error_message = error_message
        self.error_detail = error_detail
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message


# Simplified DiabetesData class for the streamlit app
class DiabetesData:
    def __init__(
        self,
        Pregnancies: float,
        BloodPressure: float,
        SkinThickness: float,
        DiabetesPedigreeFunction: float,
        Age: float,
        BMI: float = None,
        Insulin: float = None,
        Glucose: float = None,
        NewBMI: str = None,
        NewInsulinScore: str = None,
        NewGlucose: str = None
    ):
        try:
            self.Pregnancies = Pregnancies
            self.BloodPressure = BloodPressure
            self.SkinThickness = SkinThickness
            self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
            self.Age = Age
            
            # Handle BMI input (either numeric or category)
            self.BMI = BMI
            self.NewBMI = NewBMI
            
            # Handle Insulin input (either numeric or category)
            self.Insulin = Insulin
            self.NewInsulinScore = NewInsulinScore
            
            # Handle Glucose input (either numeric or category)
            self.Glucose = Glucose
            self.NewGlucose = NewGlucose
            
        except Exception as e:
            raise MyException(e, sys)

    def _infer_bmi_from_category(self, category):
        """Infer a reasonable BMI value from category if numeric value not provided"""
        bmi_mapping = {
            'Underweight': 18.0,
            'Normal': 22.0,
            'Overweight': 27.5,
            'Obesity_type1': 32.5,
            'Obesity_type2': 37.5
        }
        return bmi_mapping.get(category, 25.0)  # Default to 25 if category not found
    
    def _infer_insulin_from_category(self, category):
        """Infer a reasonable Insulin value from category if numeric value not provided"""
        insulin_mapping = {
            'Normal': 100.0,
            'Abnormal': 200.0
        }
        return insulin_mapping.get(category, 120.0)  # Default to 120 if category not found
    
    def _infer_glucose_from_category(self, category):
        """Infer a reasonable Glucose value from category if numeric value not provided"""
        glucose_mapping = {
            'Low': 65.0,
            'Normal': 85.0,
            'Overweight': 110.0,
            'Secret': 160.0,
            'High': 220.0
        }
        return glucose_mapping.get(category, 100.0)  # Default to 100 if category not found

    def get_diabetes_input_data_frame(self) -> pd.DataFrame:
        try:
            diabetes_input_dict = self.get_diabetes_data_as_dict()
            return pd.DataFrame(diabetes_input_dict)
        except Exception as e:
            raise MyException(e, sys)

    def get_diabetes_data_as_dict(self):
        try:
            # Handle BMI - prefer numeric value if provided, otherwise infer from category
            if self.BMI is None and self.NewBMI is not None:
                self.BMI = self._infer_bmi_from_category(self.NewBMI)
            elif self.BMI is None:
                raise ValueError("Either BMI value or NewBMI category must be provided")
                
            # Handle Insulin - prefer numeric value if provided, otherwise infer from category
            if self.Insulin is None and self.NewInsulinScore is not None:
                self.Insulin = self._infer_insulin_from_category(self.NewInsulinScore)
            elif self.Insulin is None:
                raise ValueError("Either Insulin value or NewInsulinScore category must be provided")
                
            # Handle Glucose - prefer numeric value if provided, otherwise infer from category
            if self.Glucose is None and self.NewGlucose is not None:
                self.Glucose = self._infer_glucose_from_category(self.NewGlucose)
            elif self.Glucose is None:
                raise ValueError("Either Glucose value or NewGlucose category must be provided")
            
            # If categories not provided, compute them from numeric values
            if self.NewBMI is None:
                self.NewBMI = self._compute_bmi_category(self.BMI)
                
            if self.NewInsulinScore is None:
                self.NewInsulinScore = self._compute_insulin_score(self.Insulin)
                
            if self.NewGlucose is None:
                self.NewGlucose = self._compute_glucose_category(self.Glucose)
            
            input_data = {
                "Pregnancies": [self.Pregnancies],
                "BloodPressure": [self.BloodPressure],
                "SkinThickness": [self.SkinThickness],
                "DiabetesPedigreeFunction": [self.DiabetesPedigreeFunction],
                "Age": [self.Age],
                "BMI": [self.BMI],
                "Insulin": [self.Insulin],
                "Glucose": [self.Glucose],
                "NewBMI": [self.NewBMI],
                "NewInsulinScore": [self.NewInsulinScore],
                "NewGlucose": [self.NewGlucose]
            }
            
            return input_data

        except Exception as e:
            raise MyException(e, sys)
            
    def _compute_bmi_category(self, bmi: float) -> str:
        """Calculate BMI category based on BMI value"""
        if bmi <= 18.5:
            return 'Underweight'
        elif bmi <= 25:
            return 'Normal'
        elif bmi <= 30:
            return 'Overweight'
        elif bmi <= 35:
            return 'Obesity_type1'
        else:
            return 'Obesity_type2'
    
    def _compute_insulin_score(self, insulin: float) -> str:
        """Calculate insulin score based on insulin value"""
        if 70 <= insulin <= 130:
            return 'Normal'
        else:
            return 'Abnormal'
    
    def _compute_glucose_category(self, glucose: float) -> str:
        """Calculate glucose category based on glucose value"""
        if glucose < 70:
            return 'Low'
        elif glucose <= 99:
            return 'Normal'
        elif glucose <= 125:
            return 'Overweight'
        elif glucose <= 200:
            return 'Secret'
        else:
            return 'High'


# Streamlit S3 model loader
class StreamlitModelLoader:
    def __init__(self):
        # Get AWS credentials from environment variables (set in Streamlit Cloud)
        self.aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.region = os.environ.get('AWS_REGION', 'us-east-1')
        
    def load_model_from_s3(self, bucket_name, model_s3_key):
        """Load model from S3 bucket"""
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.region
            )
            
            response = s3_client.get_object(Bucket=bucket_name, Key=model_s3_key)
            model_obj = pickle.loads(response['Body'].read())
            return model_obj
            
        except Exception as e:
            st.error(f"Error loading model from S3: {str(e)}")
            raise MyException(e, sys)

    def predict(self, dataframe):
        """Make prediction using the loaded model"""
        try:
            bucket_name = os.environ.get('MODEL_BUCKET_NAME')
            model_s3_key = os.environ.get('MODEL_PUSHER_S3_KEY')
            
            model = self.load_model_from_s3(bucket_name, model_s3_key)
            prediction = model.predict(dataframe)
            
            # Convert prediction to user-friendly output
            result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
            return result
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            raise MyException(e, sys)
        
# Load CSS
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

def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
        css = f"""
        <style>
        body {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)


# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Diabetes Prediction App",
        page_icon="🩺",
        layout="wide"
    )
    set_background_image("static/images/bg.jpg")
    load_css()
    show_banner()
    show_logo()

    
    st.title("Diabetes Prediction Tool")
    st.write("Enter patient information to predict diabetes risk")
    
    # Check if AWS credentials are set
    if not os.environ.get('AWS_ACCESS_KEY_ID') or not os.environ.get('AWS_SECRET_ACCESS_KEY'):
        st.warning("⚠️ AWS credentials not found. Please set them in Streamlit Cloud secrets.")
        st.info("Learn more about setting secrets in Streamlit Cloud [here](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)")
        return
    
    # Check if model info is set
    if not os.environ.get('MODEL_BUCKET_NAME') or not os.environ.get('MODEL_PUSHER_S3_KEY'):
        st.warning("⚠️ Model information not found. Please set MODEL_BUCKET_NAME and MODEL_PUSHER_S3_KEY in Streamlit Cloud secrets.")
        return
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    
    with col2:
        st.subheader("Measurements")
        input_mode = st.radio("Input Method for BMI, Insulin, and Glucose", ["Numeric Values", "Categories"])
        
        if input_mode == "Numeric Values":
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=25.0, format="%.1f")
            insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=80.0, format="%.1f")
            glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=90)
            
            new_bmi = None
            new_insulin_score = None
            new_glucose = None
        else:
            bmi = None
            insulin = None 
            glucose = None
            
            new_bmi = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obesity_type1", "Obesity_type2"])
            new_insulin_score = st.selectbox("Insulin Level", ["Normal", "Abnormal"])
            new_glucose = st.selectbox("Glucose Level", ["Low", "Normal", "Overweight", "Secret", "High"])
    
    # Prediction button
    if st.button("Predict"):
        try:
            with st.spinner("Analyzing data..."):
                # Process inputs
                diabetes_data = DiabetesData(
                    Pregnancies=pregnancies,
                    BloodPressure=blood_pressure,
                    SkinThickness=skin_thickness,
                    DiabetesPedigreeFunction=dpf,
                    Age=age,
                    BMI=bmi,
                    Insulin=insulin,
                    Glucose=glucose,
                    NewBMI=new_bmi,
                    NewInsulinScore=new_insulin_score,
                    NewGlucose=new_glucose
                )
                
                # Get dataframe for prediction
                input_df = diabetes_data.get_diabetes_input_data_frame()
                
                # Make prediction
                model_loader = StreamlitModelLoader()
                prediction = model_loader.predict(input_df)
                
                # Show result
                st.subheader("Prediction Result")
                
                if prediction == "Diabetic":
                    st.error(f"⚠️ Patient is: {prediction}. Please consult your doctor🩺.")
                else:
                    st.success(f"✅ Patient is: {prediction}. Keep up the healthy lifestyle!")
                
                # Display input data for verification
                st.subheader("Input Data Summary")
                st.write(input_df)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()