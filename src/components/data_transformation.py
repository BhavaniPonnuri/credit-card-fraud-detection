import os
import sys
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from src.utils import save_object

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for Data Transformation
        '''
        
        try:
            pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("SMOTE", SMOTE(random_state=42)),
                ]
            )
            
            return pipeline
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessor_obj = self.get_data_transformer_object()
            
            target_column = "Class"
            
            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]
            
            
            logging.info(f"Applying preprocessing object on training dataframe")
            
            train_arr_transform = preprocessor_obj.fit(input_feature_train_df, target_feature_train_df)
            
            scaler = StandardScaler()
            test_arr_transform = scaler.transform(input_feature_test_df, target_feature_test_df)
            
            train_arr = np.c_[
                train_arr_transform, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                test_arr_transform, np.array(target_feature_test_df)
            ]
            
            
            logging.info(f"Saving preprocessor object")
            
            save_object(#This is calling the function from utils.py
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj # We are saving this pickle name in the hard disk
                
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
        except Exception as e:
            raise CustomException(e, sys)
    
