from src.exception import customException
from src.logger import logging
import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split




@dataclass 
class IngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = IngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df=pd.read_csv(r'D:\Data_Science\7-Machine_Learning\projects\END-TO-END_projectsML\Customer Churn Prediction_END_TO_END_ML\NoteBook\Churn.csv')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)


            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path,)



        except Exception as e:
            raise customException(e, sys)