import os
import logging
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #defining the name format for log files
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE) #creating a full path to where the log file should be saved
os.makedirs(logs_path,exist_ok=True) #creating directories recursively without duplicates(already existing)

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE) #combines logs directory and filename to get complete file path

logging.basicConfig(
    filename=LOG_FILE_PATH, #write logs to this file
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
    
)
