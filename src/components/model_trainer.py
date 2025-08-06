import os
import sys

from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier
)

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    accuracy_score,
    confusion_matrix, 
    precision_score,
    recall_score,
    f1_score
    )
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split Train and Test data")
            
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(),
                "CatBoost Classifier":CatBoostClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "LightBoost Classifier": LGBMClassifier()
            }
            
            params = {
                "Random Forest": {
                    'n_estimators':[50, 100, 150],
                    'n_jobs':[2,4],
                    'criterion':['gini','entropy'], 
                    'verbose':[False]
                    
                },
                "XGBClassifier": {},
                "CatBoost Classifier":{
                    'iterations':[300,500], 
                    'learning_rate':[0.02, 0.01],
                    'depth':[10,12], 
                    'eval_metric':['AUC'],
                    'bagging_temperature': [0.2, 0.1, 0.3], 
                    'od_type':['Iter'], 
                    'od_wait':[100, 150, 200],
                    'verbose':[False]
                },
                "AdaBoost Classifier":{
                    
                    'learning_rate':[0.8,0.5,0.2], 
                    'n_estimators':[100, 50, 150],
                    'algorithm':['SAMME.R'], 
                    'random_state':[42, 35]
                },
                "LightBoost Classifier":{}
            }
            
            acc_report, prec_report, recall_report, f1_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                                models=models, param=params)
            
            #To get the best model from dict
            
            best_model_name = max(f1_report, key=f1_report.get)
            best_model_score = f1_report[best_model_name]
            best_model_metrics = {
                "accuarcy": acc_report[best_model_name],
                "precision": prec_report[best_model_name],
                "recall": recall_report[best_model_name],
                "f1_score": f1_report[best_model_name]
            }
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_name
            )
            
            predicted = best_model.predict(X_test)
            
            f1_score_test = f1_score(y_test, predicted)
            return f1_score_test
            
        except Exception as e:
            raise CustomException(e, sys) 