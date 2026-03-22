import sys
import os

import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            ROOT_DIR = os.getcwd()
            
            #model_path = 'artifacts\model.pkl'
            # preprocessor_path = 'artifacts\preprocessor.pkl'
            
            model_path = os.path.join(ROOT_DIR, 'artifacts', 'model.pkl')
            preprocessor_path = os.path.join(ROOT_DIR, 'artifacts', 'preprocessor.pkl')
            
            model = load_object(file_path=model_path)
            print("Type of model:", type(model))
            print("Model value:", model)
            preprocessor =load_object(file_path=preprocessor_path)
            print("Type of preprocessor:", type(preprocessor))
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            return preds
            
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Time:int,
                 V1:int, V2:int, V3:int, V4:int, V5:int, V6:int, V7:int, V8:int,
                 V9:int, V10:int, V11:int, V12:int, V13:int, V14:int, V15:int,
                 V16:int, V17:int, V18:int, V19:int, V20:int, V21:int, V22:int,
                 V23:int, V24:int, V25:int, V26:int, V27:int, V28:int, Amount:int):
        
        self.Time = Time
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.V4 = V4
        self.V5 = V5
        self.V6 = V6
        self.V7 = V7
        self.V8 = V8
        self.V9 = V9
        self.V10 = V10
        self.V11 = V11
        self.V12 = V12
        self.V13 = V13
        self.V14 = V14
        self.V15 = V15
        self.V16 = V16
        self.V17 = V17
        self.V18 = V18
        self.V19 = V19
        self.V20 = V20
        self.V21 = V21
        self.V22 = V22
        self.V23 = V23
        self.V24 = V24
        self.V25 = V25
        self.V26 = V26
        self.V27 = V27
        self.V28 = V28
        self.Amount = Amount
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Time": [self.Time],
                "V1": [self.V1],
                "V2":[self.V2],
                "V3":[self.V3],
                "V4":[self.V4],
                "V5":[self.V5],
                "V6":[self.V6],
                "V7":[self.V7],
                "V8":[self.V8],
                "V9":[self.V9],
                "V10":[self.V10],
                "V11":[self.V11],
                "V12":[self.V12],
                "V13":[self.V13],
                "V14":[self.V14],
                "V15":[self.V15],
                "V16":[self.V16],
                "V17":[self.V17],
                "V18":[self.V18],
                "V19":[self.V19],
                "V20":[self.V20],
                "V21":[self.V21],
                "V22":[self.V22],
                "V23":[self.V23],
                "V24":[self.V24],
                "V25":[self.V25],
                "V26":[self.V26],
                "V27":[self.V27],
                "V28":[self.V28],
                "Amount":[self.Amount]
                }
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)