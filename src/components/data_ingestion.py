"""
DataIngestion.py file aims to provide a simple way to read data from various sourceses. 
Here we write a code which takes the data from a source then creates a folder name artifacts apply
train_test_split on the data and save the raw_data.csv, train_data.csv, test_data.csv in the artifacts folder.
"""

## Importing libraries
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dataclasses import dataclass ## dataclass enables us to write python Class with __init__().

@dataclass ## we have to define this to intiate the class.

## Defining a class DataIngestionConfig where we define the path for the rawdata.csv, traindata.csv, testdata.csv
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifact', "train_data.csv")
    test_data_path:str = os.path.join('artifact', "test_data.csv")
    raw_data_path:str = os.path.join('artifact', "raw_data.csv")

## Now the DataIngestion components knows where to save the raw, train, test data.

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() ## The path of train_data, test_data, raw_data is stored in the ingestion_config object.

    def initiate_data_ingestion(self):
        try:
            logging.info("Entered the data for ingestion method")
            
            df = pd.read_csv("notebook\data\data.csv") ## Reading the data from the data folder.

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) ## Creating the artifacts folder if it does not exist.

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) ## Saving the raw data in the artifacts folder.

            logging.info("Train Test Split initiated")

            ## Preforming train_test_split

            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

            ## Saving the train and test data in the artifacts folder.
            train_set.to_csv(self.ingestion_config.train_data_path)
            test_set.to_csv(self.ingestion_config.test_data_path)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

            





