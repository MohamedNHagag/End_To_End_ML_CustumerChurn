import os 
import sys
import numpy as np
from src.exception import customException
from src.logger import logging
from src.components.transformation import DataTransformation
from src.components.ingestion import DataIngestion
import pickle



def save_object(file_path, obj):
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)



def load_object(file_path):
      with open(file_path, 'rb') as file:
            return pickle.load(file)

