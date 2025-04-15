from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from uvicorn import run as app_run

from typing import Optional

# Constants and pipeline imports
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import DiabetesData,DiabetesDataClassifier
from src.pipline.training_pipeline import TrainPipeline

# -------------------- APP INIT --------------------
app = FastAPI()

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

# CORS setup
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- FORM CLASS --------------------
class DiabetesForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Pregnancies: Optional[int] = None
        self.BloodPressure: Optional[float] = None
        self.SkinThickness: Optional[float] = None
        self.DiabetesPedigreeFunction: Optional[float] = None
        self.Age: Optional[int] = None
        
        # Numeric values that will be inferred if not provided
        self.BMI: Optional[float] = None
        self.Insulin: Optional[float] = None
        self.Glucose: Optional[float] = None
        
        # Dropdown values
        self.NewBMI: Optional[str] = None
        self.NewInsulinScore: Optional[str] = None
        self.NewGlucose: Optional[str] = None

    async def get_diabetes_data(self):
        form = await self.request.form()
        self.Pregnancies = float(form.get("Pregnancies"))
        self.BloodPressure = float(form.get("BloodPressure"))
        self.SkinThickness = float(form.get("SkinThickness"))
        self.DiabetesPedigreeFunction = float(form.get("DiabetesPedigreeFunction"))
        self.Age = int(form.get("Age"))

        # Try to get numeric values if provided, otherwise they'll remain None
        try:
            if form.get("BMI") and form.get("BMI").strip():
                self.BMI = float(form.get("BMI"))
        except (ValueError, TypeError):
            pass
            
        try:
            if form.get("Insulin") and form.get("Insulin").strip():
                self.Insulin = float(form.get("Insulin"))
        except (ValueError, TypeError):
            pass
            
        try:
            if form.get("Glucose") and form.get("Glucose").strip():
                self.Glucose = float(form.get("Glucose"))
        except (ValueError, TypeError):
            pass

        # Get categorical values from dropdowns
        self.NewBMI = form.get("NewBMI")
        self.NewInsulinScore = form.get("NewInsulinScore")
        self.NewGlucose = form.get("NewGlucose")

# -------------------- ROUTES --------------------
@app.get("/", tags=["UI"])
async def index(request: Request):
    bmi_options = ['Underweight', 'Normal', 'Overweight', 'Obesity_type1', 'Obesity_type2']
    insulin_options = ['Normal', 'Abnormal']
    glucose_options = ['Low', 'Normal', 'Overweight', 'Secret', 'High']
    
    return templates.TemplateResponse(
        "diabetesform.html", 
        {
            "request": request, 
            "context": "Rendering",
            "bmi_options": bmi_options,
            "insulin_options": insulin_options,
            "glucose_options": glucose_options
        }
    )

@app.get("/train", tags=["Training"])
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("✅ Training successful!")
    except Exception as e:
        return Response(f"❌ Error occurred during training: {e}")

@app.post("/", tags=["Prediction"])
async def predictRouteClient(request: Request):
    try:
        # Get form data
        form = DiabetesForm(request)
        await form.get_diabetes_data()
        
        # BMI dropdown options for response context
        bmi_options = ['Underweight', 'Normal', 'Overweight', 'Obesity_type1', 'Obesity_type2']
        insulin_options = ['Normal', 'Abnormal']
        glucose_options = ['Low', 'Normal', 'Overweight', 'Secret', 'High']

        # Create DiabetesData object with all available inputs
        user_data = DiabetesData(
            Pregnancies=form.Pregnancies,
            BloodPressure=form.BloodPressure,
            SkinThickness=form.SkinThickness,
            DiabetesPedigreeFunction=form.DiabetesPedigreeFunction,
            Age=form.Age,
            BMI=form.BMI,               # Optional numeric value
            Insulin=form.Insulin,       # Optional numeric value
            Glucose=form.Glucose,       # Optional numeric value
            NewBMI=form.NewBMI,         # Dropdown selection
            NewInsulinScore=form.NewInsulinScore,  # Dropdown selection
            NewGlucose=form.NewGlucose  # Dropdown selection
        )

        # Get input dataframe with all features
        diabetes_df = user_data.get_diabetes_input_data_frame()

        # Initialize the prediction pipeline
        model_predictor = DiabetesDataClassifier()

        # Make a prediction and retrieve the result
        value = model_predictor.predict(dataframe=diabetes_df)[0]
        
        # Format result for display
        status = "🩺 Patient is **Diabetic**" if value==1 else "✅ Patient is **Non-Diabetic**"

        return templates.TemplateResponse(
            "diabetesform.html", 
            {
                "request": request, 
                "context": status,
                "bmi_options": bmi_options,
                "insulin_options": insulin_options,
                "glucose_options": glucose_options
            }
        )

    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        return templates.TemplateResponse(
            "diabetesform.html", 
            {
                "request": request, 
                "context": error_message,
                "bmi_options": ['Underweight', 'Normal', 'Overweight', 'Obesity_type1', 'Obesity_type2'],
                "insulin_options": ['Normal', 'Abnormal'],
                "glucose_options": ['Low', 'Normal', 'Overweight', 'Secret', 'High']
            }
        )

# -------------------- MAIN --------------------
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)