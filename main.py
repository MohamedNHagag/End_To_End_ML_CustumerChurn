from src.components.ingestion import DataIngestion
from src.components.transformation import DataTransformation
from src.components.trainer import ModelTrainer
from src.pipeline.prediction import CustomerData
from src.utils import save_object, load_object
from src.logger import logging
from src.exception import customException
import sys





if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        x_train, y_train, x_test, y_test = data_transformation.initiate_model(train_path, test_path)

        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(x_train, y_train, x_test, y_test)

        print(f"Model trained with accuracy: {accuracy}")


    except Exception as e:
        raise customException(e, sys)


