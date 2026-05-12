import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)#dill is used to save the object in binary format

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]#to get the model object from the dict
            model.fit(X_train,y_train)#to train the model
            
            y_train_pred=model.predict(X_train)#to get the predicted values for the training data
            y_test_pred=model.predict(X_test)#to get the predicted values for the test data
            
            train_model_score=r2_score(y_train,y_train_pred)#to get the r2 score for the training data
            test_model_score=r2_score(y_test,y_test_pred)#to get the r2 score for the test data

            report[list(models.keys())[i]]=test_model_score#to get the model name from the dict and to get the test score for the model and to save it in the report dict

        return report

    except Exception as e:
        raise CustomException(e,sys)