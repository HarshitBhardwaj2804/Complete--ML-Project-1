"""
In this file we load our train, test array's that we created using the initiate_data_transformation function inside the 
DataTransformation class of data_transformation.py file.
We import multiple regression models(as the problem is regression problem) train them on X_train, y_train find the best model
and then save it in the artifact folder by name model.pkl .
"""

## Importing necessary libraries
import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import train_model, save_object

@dataclass
class ModelTrainerConfig:
    """Model Trainer Config class to store the configuration of model trainer."""
    trainer_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            """
            1. Break train_array into X_train and y_train, test_array into X_test and y_test.
            2. Initailze the models
            3. Train the models on X_train, y_train
            4. Find the best model and save it in the artifact folder as model.pkl.
            """

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], ## give me all rows, all columns except last one from train_array.
                train_array[:, -1], ## give me all rows, last column from train_array.
                test_array[:,:-1], ## give me all rows, all columns except last one from test_array.
                test_array[:,-1] ## give me all rows, last column from test_array.
            )
        
            models = {
                "LinearRegression": LinearRegression(),
                #"Ridge": Ridge(),
                #"Lasso": Lasso(),
                #"SVR": SVR(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),   
                "AdaBoostRegressor": AdaBoostRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                #"KNeighborsRegressor": KNeighborsRegressor(),
                "CatBoostingRegressor": CatBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
            }

            params={
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostingRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report: dict = train_model(X_train, y_train, X_test, y_test, models, params) ## define in untils.py

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<.60:
                raise CustomException("No best model found")

            save_object(
                file_path = self.model_trainer_config.trainer_model_file_path,
                obj =  best_model
            )

            prediction = best_model.predict(X_test)

            score = r2_score(y_test, prediction)

            return score
        
        except Exception as e:
            raise CustomException(e,sys)