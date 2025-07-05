import pandas as pd
import numpy as np  
import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import customException
from src.logger import logging
from src.components.transformation import DataTransformation
from src.components.ingestion import DataIngestion
from src.components.evaluate import evaluate_model
from src.utils import save_object, load_object






@dataclass
class TrainerConfig:
    model_trainer_path:str=os.path.join('artificts','model.pkl')


def initiate_model_trainer(self, x_train, y_train, x_test, y_test):
    try:
        logging.info("Model Training initiated")

        models = {
            'LogisticRegression': LogisticRegression(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            'CatBoostClassifier': CatBoostClassifier(verbose=0)
        }
        params = {
            'LogisticRegression': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']},
            'KNeighborsClassifier': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
            'DecisionTreeClassifier': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]},
            'RandomForestClassifier': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
            'GradientBoostingClassifier': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            'AdaBoostClassifier': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            'XGBClassifier': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            'CatBoostClassifier': {'iterations': [50, 100], 'learning_rate': [0.01, 0.1]}
        }

        evaluate_model_result = evaluate_model(x_train, y_train, x_test, y_test, models, params)
        best_model_name = max(evaluate_model_result, key=lambda x: evaluate_model_result[x]['test_accuracy'])
        best_model = models[best_model_name]

        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        save_object(self.trainer_config.model_trainer_path, best_model)
        return accuracy

    except Exception as e:
        raise customException(e, sys)

        

