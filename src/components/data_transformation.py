import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder,RobustScaler
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

    def _replace_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace zero values in specified columns with NaN for later imputation."""
        try:
            logging.info("Replacing zero values with NaN.")
            columns = self._schema_config['columns_to_replace_zeros']
            for col in columns:
                if col in df.columns:
                    df[col] = df[col].replace(0, np.nan)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def _impute_missing_values_by_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing (NaN) values in numeric columns using the mean of that column grouped by target class.
        """
        try:
            logging.info("Imputing missing values by class (Outcome).")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if TARGET_COLUMN not in df.columns:
                raise Exception(f"{TARGET_COLUMN} not found in dataframe for class-based imputation.")

            for col in numeric_cols:
                if col != TARGET_COLUMN and df[col].isnull().sum() > 0:
                    df[col] = df.groupby(TARGET_COLUMN)[col].transform(lambda x: x.fillna(x.mean()))

            return df
        except Exception as e:
            raise MyException(e, sys)
        
    def _impute_outliers_with_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace outliers in numeric columns using IQR method:
        - Values below Q1 - 1.5*IQR will be replaced with Q1
        - Values above Q3 + 1.5*IQR will be replaced with Q3
        """
        try:
            logging.info("Imputing outliers using IQR method.")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != TARGET_COLUMN]

            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                df[col] = np.where(df[col] < lower_bound, Q1, df[col])
                df[col] = np.where(df[col] > upper_bound, Q3, df[col])

            return df

        except Exception as e:
            raise MyException(e, sys)


    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply domain-specific feature engineering as per schema."""
        logging.info("Starting feature engineering...")
        # BMI Category
        df['NewBMI'] = pd.cut(
            df['BMI'],
            bins=[0, 18.5, 25, 30, 35, np.inf],
            labels=['Underweight', 'Normal', 'Overweight', 'Obesity_type1', 'Obesity_type2']
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
    


    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns as specified in the schema config under 'drop_columns',
        and log the names of the columns being dropped.
        """
        try:
            logging.info("Checking columns to drop as per schema config.")
            drop_cols = self._schema_config.get('drop_columns', [])
            if isinstance(drop_cols, str):
                drop_cols = [drop_cols]

            drop_cols_present = [col for col in drop_cols if col in df.columns]

            if drop_cols_present:
                logging.info(f"Dropping columns: {drop_cols_present}")
                df = df.drop(columns=drop_cols_present)
            else:
                logging.info("No matching columns found to drop.")

            return df

        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            robust_scaler = RobustScaler()
            standard_scaler = StandardScaler()
            one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            logging.info("Transformers Initialized: RobustScaler, StandardScaler, OneHotEncoder")

            # Load schema configurations
            robust_scaler_columns = self._schema_config['robust_scaler_columns']
            standard_scaler_columns = self._schema_config['standard_scaler_columns']
            ohe_columns = self._schema_config['columns_to_apply_one_hot_encoding']

            logging.info("Columns loaded from schema")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("RobustScaler", robust_scaler, robust_scaler_columns),
                    ("StandardScaler", standard_scaler, standard_scaler_columns),
                    ("OneHotEncoder", one_hot_encoder, ohe_columns)
                ],
                remainder='passthrough'
            )

            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object")
            raise MyException(e, sys)


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Replace zeros
            train_df = self._replace_zeros(train_df)
            test_df = self._replace_zeros(test_df)

            # Impute missing values with class-wise mean
            train_df = self._impute_missing_values_by_class(train_df)
            test_df = self._impute_missing_values_by_class(test_df)

            # Impute outliers using IQR method
            train_df = self._impute_outliers_with_iqr(train_df)
            test_df = self._impute_outliers_with_iqr(test_df)

            # Feature engineering
            train_df = self._feature_engineering(train_df)
            test_df = self._feature_engineering(test_df)

            # Drop columns based on schema
            train_df = self._drop_columns(train_df)
            test_df = self._drop_columns(test_df)

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Apply transformations
            preprocessor = self.get_data_transformer_object()
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTEENN applied to train-test df.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e