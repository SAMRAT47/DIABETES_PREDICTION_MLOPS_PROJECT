import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
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

    def get_data_transformer_object(self) -> Pipeline:
        """Creates and returns a data transformer object for the data."""
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            # Initialize transformers
            robust_scaler = RobustScaler()
            # standard_scaler = StandardScaler()
            logging.info("Transformers Initialized: RobustScaler-StandardScaler")
            
            #Load schema configurations
            robust_columns = self._schema_config['robust_columns']
            # standard_columns = self._schema_config['std_columns']
            logging.info("Cols loaded from schema.")
            
            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("RobustScaler", robust_scaler, robust_columns)
                    # ,
                    # ("StandardScaler", standard_scaler, standard_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )
            
            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline
        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e



    def _replace_zeros_with_nan(self, df):
        """Replace zeros with NaN in columns except the target column and log NaN values information."""
        logging.info("Replacing zeros with NaN in columns except the target column")
        columns_to_replace = [col for col in df.columns if col != TARGET_COLUMN]
        df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)

        # Log NaN values information
        nan_counts = df[columns_to_replace].isnull().sum()
        logging.info("NaN values information after replacement:")
        for col, count in nan_counts.items():
            logging.info(f"Column: {col}, NaN count: {count}, NaN percentage: {(count / len(df)) * 100:.2f}%")

        return df
    
    def _replace_nan_with_class_mean(self, input_feature_train_df, target_feature_train_df):
        logging.info("Entering _replace_nan_with_class_mean method")
        
        target_column = TARGET_COLUMN
        
        temp_df = pd.concat([input_feature_train_df, target_feature_train_df], axis=1)
        
        for col in temp_df.columns:
            if col != target_column and temp_df[col].dtype.kind in 'bifc':  # Check if column is numeric
                mean_values = temp_df[[col, target_column]].dropna(subset=[col]).groupby(target_column)[col].mean().reset_index()
                for class_value in temp_df[target_column].unique():
                    temp_df.loc[(temp_df[target_column] == class_value) & (temp_df[col].isnull()), col] = round(mean_values.loc[mean_values[target_column] == class_value, col].values[0])
        
        input_feature_train_df = temp_df.drop(columns=[target_column])
        
        logging.info("Exiting _replace_nan_with_class_mean method")
        
        return input_feature_train_df



    def _replace_outliers_with_q1_q3(self, df):
        """Detect outliers and replace them with Q1 and Q3 values respectively."""
        logging.info("Detecting outliers and replacing them with Q1 and Q3 values respectively")

        # Detect outliers and replace them with Q1 and Q3 values respectively
        for col in df.columns:
            if col != TARGET_COLUMN:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].apply(lambda x: Q1 if x < lower_bound else (Q3 if x > upper_bound else x))

        # Log information about outliers
        logging.info("Outliers replaced with Q1 and Q3 values:")
        for col in df.columns:
            if col != TARGET_COLUMN:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
                logging.info(f"Column: {col}, Outlier count after replacement: {outlier_count}")

        return df


    def _create_new_bmi_column(self, df):
        """Create a new column named 'NewBMI' based on the ranges of the 'BMI' column."""
        logging.info("Creating a new column named 'NewBMI' based on the ranges of the 'BMI' column")
        bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
        labels = ["Underweight","Normal", "Overweight","Obesity 1", "Obesity 2", "Obesity 3"]
        df['NewBMI'] = pd.cut(df['BMI'], bins=bins, labels=labels, right=False)
        logging.info("New column 'NewBMI' created successfully")
        return df

    def _create_new_insulin_score_column(self, df):
        """Create a new column named 'NewInsulinScore' based on the 'Insulin' column."""
        logging.info("Creating a new column named 'NewInsulinScore' based on the 'Insulin' column")
        df['NewInsulinScore'] = np.where((df['Insulin'] >= 16) & (df['Insulin'] <= 166), 'Normal', 'Abnormal')
        logging.info("New column 'NewInsulinScore' created successfully")
        return df
    
    def _create_new_glucose_column(self, df):
        """Create a new column named 'NewGlucose' based on the 'Glucose' column."""
        logging.info("Creating a new column named 'NewGlucose' based on the 'Glucose' column")
        bins = [0, 70, 99, 126, float('inf')]
        labels = ["Low", "Normal", "Overweight", "Secret"]
        df['NewGlucose'] = pd.cut(df['Glucose'], bins=bins, labels=labels, right=False)
        logging.info("New column 'NewGlucose' created successfully")
        return df

    def _apply_one_hot_encoding(self, df):
        """Apply one-hot encoding to the specified columns."""
        logging.info("Applying one-hot encoding to the specified columns")
        columns_to_encode = ["NewBMI", "NewInsulinScore", "NewGlucose"]
        df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True).astype(int)
        logging.info("One-hot encoding applied successfully")
        return df



    # def _create_dummy_columns(self, df):
    #     """Create dummy variables for categorical features."""
    #     logging.info("Creating dummy variables for categorical features")
    #     df = pd.get_dummies(df, drop_first=True)
    #     return df

    # def _rename_columns(self, df):
    #     """Rename specific columns and ensure integer types for dummy columns."""
    #     logging.info("Renaming specific columns and casting to int")
    #     df = df.rename(columns={
    #         "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
    #         "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
    #     })
    #     for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
    #         if col in df.columns:
    #             df[col] = df[col].astype('int')
    #     return df

    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations in specified sequence
            transformation_functions = [
                self._drop_id_column,
                self._replace_zeros_with_nan,
                self._replace_outliers_with_q1_q3,
                self._create_new_bmi_column,
                self._create_new_insulin_score_column,
                self._create_new_glucose_column,
                self._apply_one_hot_encoding
            ]

            for func in transformation_functions:
                input_feature_train_df = func(input_feature_train_df)
                input_feature_test_df = func(input_feature_test_df)

                input_feature_train_df = self._replace_nan_with_class_mean(input_feature_train_df, target_feature_train_df)
                input_feature_test_df = self._replace_nan_with_class_mean(input_feature_test_df, target_feature_test_df)
                logging.info("Custom transformations applied to train and test data")
                
                # Get the preprocessor object
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                print(input_feature_train_df.columns)
                print(input_feature_test_df.columns)

                
                # Apply transformation to training and testing data
                logging.info("Initializing transformation for Training-data")
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                logging.info("Initializing transformation for Testing-data")
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                logging.info("Transformation done end to end to train-test df.")

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