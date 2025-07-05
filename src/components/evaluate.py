import os 
import sys
import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.exception import customException
from src.logger import logging
from src.components.transformation import DataTransformation
from src.components.ingestion import DataIngestion
from sklearn.model_selection import RandomizedSearchCV


def evaluate_model(train_X,train_y,test_x,test_y,models:dict,params:dict):
        try:
                evaluate_model = {}
                for name,model_obj in models.items():
                    if name in params and params[name]:
                        randomsearch=RandomizedSearchCV(
                            estimator=model_obj,
                            param_distributions=params[name],
                            cv=3,
                            n_iter=10,
                            verbose=2,
                            n_jobs=-1)
                        randomsearch.fit(train_X, train_y)
                        best_model=randomsearch.best_estimator_
            
            
                    else:
                        best_model=model_obj
                        best_model.fit(train_X, train_y)
            
                    predict_train=best_model.predict(train_X)
                    predict_test=best_model.predict(test_x)
            
                    train_accuracy=accuracy_score(train_y,predict_train)
                    test_accuracy=accuracy_score(test_y,predict_test) 
                    train_precision=precision_score(train_y,predict_train,average='weighted')
                    test_precision=precision_score(test_y,predict_test,average='weighted')
                    train_recall=recall_score(train_y,predict_train,average='weighted')
                    test_recall=recall_score(test_y,predict_test,average='weighted')
                    train_f1=f1_score(train_y,predict_train,average='weighted')
                    test_f1=f1_score(test_y,predict_test,average='weighted')
            
                    evaluate_model[name]={
                        'train_accuracy': train_accuracy, 
                        'test_accuracy': test_accuracy,
                        'train_precision': train_precision,
                        'test_precision': test_precision,
                        'train_recall': train_recall,
                        'test_recall': test_recall,
                        'train_f1': train_f1,
                        'test_f1': test_f1}
                    
            
                return evaluate_model


        except Exception as e:
                raise customException(e, sys)