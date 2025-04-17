import sys
import pickle
import boto3
import numpy as np
import pandas as pd
from src.entity.config_entity import DiabetesPredictorConfig
from src.exception import MyException
from src.logger import logging


def load_object(data_bytes: bytes):
    try:
        return pickle.loads(data_bytes)
    except Exception as e:
        raise MyException(f"Failed to deserialize model object: {e}", sys)


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
            self.BMI = BMI
            self.NewBMI = NewBMI
            self.Insulin = Insulin
            self.NewInsulinScore = NewInsulinScore
            self.Glucose = Glucose
            self.NewGlucose = NewGlucose
        except Exception as e:
            raise MyException(e, sys) from e

    def _infer_bmi_from_category(self, category):
        bmi_mapping = {
            'Underweight': 18.0,
            'Normal': 22.0,
            'Overweight': 27.5,
            'Obesity_type1': 32.5,
            'Obesity_type2': 37.5
        }
        return bmi_mapping.get(category, 25.0)

    def _infer_insulin_from_category(self, category):
        insulin_mapping = {
            'Normal': 100.0,
            'Abnormal': 200.0
        }
        return insulin_mapping.get(category, 120.0)

    def _infer_glucose_from_category(self, category):
        glucose_mapping = {
            'Low': 65.0,
            'Normal': 85.0,
            'Overweight': 110.0,
            'Secret': 160.0,
            'High': 220.0
        }
        return glucose_mapping.get(category, 100.0)

    def _compute_bmi_category(self, bmi: float) -> str:
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
        if 70 <= insulin <= 130:
            return 'Normal'
        else:
            return 'Abnormal'

    def _compute_glucose_category(self, glucose: float) -> str:
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

    def get_diabetes_data_as_dict(self):
        try:
            logging.info("Creating input dictionary from provided values")

            if self.BMI is None and self.NewBMI is not None:
                self.BMI = self._infer_bmi_from_category(self.NewBMI)
            elif self.BMI is None:
                raise ValueError("Either BMI value or NewBMI category must be provided")

            if self.Insulin is None and self.NewInsulinScore is not None:
                self.Insulin = self._infer_insulin_from_category(self.NewInsulinScore)
            elif self.Insulin is None:
                raise ValueError("Either Insulin value or NewInsulinScore must be provided")

            if self.Glucose is None and self.NewGlucose is not None:
                self.Glucose = self._infer_glucose_from_category(self.NewGlucose)
            elif self.Glucose is None:
                raise ValueError("Either Glucose value or NewGlucose must be provided")

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

    def get_diabetes_input_data_frame(self) -> pd.DataFrame:
        try:
            return pd.DataFrame(self.get_diabetes_data_as_dict())
        except Exception as e:
            raise MyException(e, sys) from e


class DiabetesDataClassifier:
    def __init__(self, bucket_name: str = None, model_path: str = None,
                 aws_access_key_id: str = None, aws_secret_access_key: str = None) -> None:
        """
        :param bucket_name: S3 bucket name
        :param model_path: Path to the model in S3
        :param aws_access_key_id: AWS Access Key ID
        :param aws_secret_access_key: AWS Secret Access Key
        """
        try:
            self.bucket_name = bucket_name
            self.model_path = model_path
            self.aws_access_key_id = aws_access_key_id
            self.aws_secret_access_key = aws_secret_access_key

            # Initialize S3 client
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )

        except Exception as e:
            raise MyException(e, sys)

    def load_model_from_s3(self):
        try:
            logging.info(f"Loading model from S3: {self.bucket_name}/{self.model_path}")
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=self.model_path)
            data = obj['Body'].read()
            return load_object(data)
        except Exception as e:
            raise MyException(f"Error loading model from S3: {e}", sys) from e

    def predict(self, dataframe: pd.DataFrame) -> str:
        try:
            logging.info("Entered predict method of DiabetesDataClassifier class")
            model = self.load_model_from_s3()
            result = model.predict(dataframe)
            return result
        except Exception as e:
            raise MyException(e, sys)