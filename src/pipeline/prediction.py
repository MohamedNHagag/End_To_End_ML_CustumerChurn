import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.exception import customException
from src.logger import logging
from src.utils import load_object,save_object
from src.components.transformation import DataTransformation
from src.components.ingestion import DataIngestion
from src.components.evaluate import evaluate_model
from src.components.trainer import ModelTrainer



@dataclass
class CustomerData:
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

    def get_data(self):
        data = {
            "gender": self.gender,
            "SeniorCitizen": self.SeniorCitizen,
            "Partner": self.Partner,
            "Dependents": self.Dependents,
            "tenure": self.tenure,
            "PhoneService": self.PhoneService,
            "MultipleLines": self.MultipleLines,
            "InternetService": self.InternetService,
            "OnlineSecurity": self.OnlineSecurity,
            "OnlineBackup": self.OnlineBackup,
            "DeviceProtection": self.DeviceProtection,
            "TechSupport": self.TechSupport,
            "StreamingTV": self.StreamingTV,
            "StreamingMovies": self.StreamingMovies,
            "Contract": self.Contract,
            "PaperlessBilling": self.PaperlessBilling,
            "PaymentMethod": self.PaymentMethod,
            "MonthlyCharges": self.MonthlyCharges,
            "TotalCharges": self.TotalCharges
        }
        return pd.DataFrame([data])




class PredictionPipeline:
    def __init__(self):
        self.data_transformation = DataTransformation()
        self.model = load_object(os.path.join('artifacts', 'model.pkl'))

    def predict(self, data: CustomerData):
        try:
            logging.info("Prediction initiated")
            data_df = data.get_data()
            transformed_data = self.data_transformation.preprocess_data(data_df)
            prediction = self.model.predict(transformed_data)
            return prediction[0]
        except Exception as e:
            raise customException(e, sys)


