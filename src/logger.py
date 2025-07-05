from src.exception import customException
import os
import sys
import numpy as np
from datetime import datetime
import logging


file_name=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

log_file=os.path.join(os.getcwd(),"logs")
os.makedirs(log_file,exist_ok=True)

log_file_path=os.path.join(log_file,file_name+".log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)


