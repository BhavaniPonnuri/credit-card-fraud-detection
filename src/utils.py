import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    accuracy_score,
    confusion_matrix, 
    precision_score,
    recall_score,
    f1_score
    )


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        acc_report = {}
        prec_report = {}
        recall_report = {}
        f1_report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            rs = RandomizedSearchCV(model, para, cv=3)
            rs.fit(X_train,y_train)
            
            model.set_params(**rs.best_params_) #setting the best parameters
            model.fit(X_train, y_train) #Training model after tuning the parameters
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_accuarcy = accuracy_score(y_train, y_train_pred)
            train_model_precision = precision_score(y_train, y_train_pred, average='weighted')
            train_model_recall = recall_score(y_train, y_train_pred, average='weighted')
            train_model_f1_score = f1_score(y_train, y_train_pred, average='weighted')
            
            test_model_accuarcy = accuracy_score(y_test, y_test_pred)
            test_model_precision = precision_score(y_test, y_test_pred, average='weighted')
            test_model_recall = recall_score(y_test, y_test_pred, average='weighted')
            test_model_f1_score = f1_score(y_test, y_test_pred, average='weighted')
            
            acc_report[list(model.keys())[i]] = test_model_accuarcy
            prec_report[list(model.keys())[i]] = test_model_precision
            recall_report[list(model.keys())[i]] = test_model_recall
            f1_report[list(model.keys())[i]] = test_model_f1_score
            
            return acc_report, prec_report, recall_report, f1_report
            
    except Exception as e:
        raise CustomException(e, sys)