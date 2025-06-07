"""
In this python file we load our train, test data apply preprocessing steps like handling missing data, 
encoding categorical variables, applying standardization. 
For this we will use scikit-learn pipelines.
We load the data, apply preprocessing pipelines, save the pipepline as .pkl file and save the preprocessod train, test data.
"""

## Importing necessary libraries
import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """In this class we define the path at which our data_transformation
    component/object stores the final model as .pkl file."""

    preprocessor_obj_file_path = os.path.join('artifact', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()

    ## Definging a function which preforms all data preprocessing
    ## steps and then return preprocessor object as output.
    def get_data_transformer_obj(self):
        try:
            """
            1. Define numerical, categorical columns in a list.
            2. Create seprate pipeline for numerical and categorical columns.
            3. Use column transformer to combine both pipelines.
            4. return the column transformer object as ouput.
            """

            numerical_cols = ['writing_score', 'reading_score']
            categorical_cols = ["gender", "race_ethnicity", "parental_level_of_education", "lunch","test_prepration_course"]

            ## Creating Numeric Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy = "median")),
                    ("scaler", StandardScaler())
                ]
            )
            ## Creating Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("onehotencoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")

            ## Combinging both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols)
                ]
            )

            ## Returning object
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            """
            1. Load train and test data from respective paths.
            2. Load the data transformer object using the above function.
            3. Split train data into X_train, y_train & split test data into
               X_test, y_test.
            4. Apply the data-transformer object on X_train, X_test.
            5. Save the data-transformer object as .pkl in the artifact folder.
            6. Return transformerd train data, test data, object's path.
            """

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_object = self.get_data_transformer_obj()

            target_column_name = "math_score"

            X_train = train_df.drop(columns=[target_column_name],axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns = [target_column_name], axis = 1)
            y_test = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            X_train_arr = preprocessing_object.fit_transform(X_train)
            X_test_arr = preprocessing_object.transform(X_test)

            train_arr = np.c_[
                X_train_arr, np.array(y_train)
            ]
            test_arr = np.c_[
                X_test_arr, np.array(y_test)
                ]
            
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transform_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )

            return (
                train_arr,
                test_arr,
                self.data_transform_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)