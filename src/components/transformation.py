from src.exception import customException
from src.logger import logging
import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

@dataclass
class TransformationConfig:
    processor_path: str = os.path.join('artifacts', 'processor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = TransformationConfig()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Encode 'Churn'
            df['Churn'] = self.label_encoder.fit_transform(df['Churn'])

            # Encode binary columns
            binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
            for col in binary_columns:
                df[col] = df[col].replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

            # Encode multi-category columns
            multi_columns = {
                'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': -1},
                'InternetService': {'DSL': 1, 'Fiber optic': 2, 'No': 0},
                'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': -1},
                'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': -1},
                'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': -1},
                'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': -1},
                'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': -1},
                'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': -1},
                'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
                'PaymentMethod': {'Mailed check': 0,'Bank transfer (automatic)': 1,'Electronic check': 2,'Credit card (automatic)': 3
}
            }

            for col, mapping in multi_columns.items():
                df[col] = df[col].replace(mapping)

            # Scale numerical columns
            num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            df[num_cols] = self.scaler.fit_transform(df[num_cols])

            return df

        except Exception as e:
            raise customException(e, sys)

    def initiate_model(self, train_data, test_data):
        try:
            logging.info("Data Preprocessing initiated")
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            # Apply preprocessing
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            X_train = train_df.drop('Churn', axis=1)
            y_train = train_df['Churn']
            X_test = test_df.drop('Churn', axis=1)
            y_test = test_df['Churn']

            return np.array(X_train, dtype=np.float32), y_train, np.array(X_test, dtype=np.float32), y_test

        except Exception as e:
            raise customException(e, sys)
