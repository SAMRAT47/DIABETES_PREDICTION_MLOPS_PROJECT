import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def replace_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace zero values in specified columns with the median of that column."""
        columns = self._schema_config['columns_to_replace_zeros']
        for col in columns:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        return df

    def impute_missing_values_by_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in each numeric column using the mean of that column grouped by target class.
        """
        try:
            logging.info("Imputing missing values by class.")
            df_copy = df.copy()
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()

            if TARGET_COLUMN not in df_copy.columns:
                raise Exception(f"{TARGET_COLUMN} not found in dataframe for class-based imputation.")

            for col in numeric_cols:
                if df_copy[col].isnull().sum() > 0 and col != TARGET_COLUMN:
                    df_copy[col] = df_copy.groupby(TARGET_COLUMN)[col].transform(lambda x: x.fillna(x.mean()))

            return df_copy
        except Exception as e:
            raise MyException(e, sys)


    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply domain-specific feature engineering as per schema."""
        logging.info("Starting feature engineering...")
        # BMI Category
        df['NewBMI'] = pd.cut(
            df['BMI'],
            bins=[0, 18.5, 25, 30, 35, np.inf],
            labels=['Underweight', 'Normal', 'Overweight', 'Obesity 1', 'Obesity 2']
        )

        # Insulin Score
        df['NewInsulinScore'] = df['Insulin'].apply(lambda x: "Normal" if 70 <= x <= 130 else "Abnormal")

        # Glucose Category
        df['NewGlucose'] = pd.cut(
            df['Glucose'],
            bins=[0, 70, 99, 125, 200, np.inf],
            labels=['Low', 'Normal', 'Overweight', 'Secret', 'High']
        )
        logging.info("Feature engineering done.")
        return df

    def get_data_transformer_object(self):
            try:
                logging.info("Creating data transformer object.")

                num_pipeline = Pipeline(steps=[
                    ('scaler', StandardScaler())
                ])

                cat_pipeline = Pipeline(steps=[
                    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                ])

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', num_pipeline, self._schema_config['standard_scaler_columns']),
                        ('cat', cat_pipeline, self._schema_config['columns_to_apply_one_hot_encoding'])
                    ])

                logging.info("Pipeline created successfully.")
                return preprocessor

            except Exception as e:
                logging.exception("Exception occurred while creating transformer object.")
                raise MyException(e, sys) from e


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Replace zeros
            train_df = self.replace_zeros(train_df)
            test_df = self.replace_zeros(test_df)

            # Impute missing values with class-wise mean
            train_df = self.impute_missing_values_by_class(train_df)
            test_df = self.impute_missing_values_by_class(test_df)

            # Feature engineering
            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            # Drop ID column
            drop_column = self._schema_config['drop_columns']
            if drop_column in train_df.columns:
                train_df.drop(columns=[drop_column], inplace=True)
            if drop_column in test_df.columns:
                test_df.drop(columns=[drop_column], inplace=True)

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Apply transformations
            preprocessor = self.get_data_transformer_object()
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Handle imbalance
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )

            # Combine features and targets
            train_arr = np.c_[input_feature_train_final, target_feature_train_final]
            test_arr = np.c_[input_feature_test_final, target_feature_test_final]

            # Save outputs
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            logging.info("Data transformation completed successfully.")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e