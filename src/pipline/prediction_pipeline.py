import sys
import numpy as np
import pandas as pd
from src.entity.config_entity import DiabetesPredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object


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
            raise MyException(e, sys) from e

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
            raise MyException(e, sys) from e

    def get_diabetes_data_as_dict(self):
        try:
            logging.info("Creating input dictionary from provided values")
            
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
            
            logging.info(f"Created input data: {input_data}")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e
            
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


class DiabetesDataClassifier:
    def __init__(self,prediction_pipeline_config: DiabetesPredictorConfig = DiabetesPredictorConfig(),) -> None:
            """
            :param prediction_pipeline_config: Configuration for prediction the value
            """
            try:
                self.prediction_pipeline_config = prediction_pipeline_config
            except Exception as e:
                raise MyException(e, sys)

    def predict(self, dataframe) -> str:
            """
            This is the method of VehicleDataClassifier
            Returns: Prediction in string format
            """
            try:
                logging.info("Entered predict method of VehicleDataClassifier class")
                model = Proj1Estimator(
                    bucket_name=self.prediction_pipeline_config.model_bucket_name,
                    model_path=self.prediction_pipeline_config.model_file_path,
                )
                result =  model.predict(dataframe)
                
                return result
            
            except Exception as e:
                raise MyException(e, sys)