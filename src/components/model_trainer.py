import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")#used for saving the model after training

class ModelTrainer:#used for training the model
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],#all the columns except the last one
                train_array[:,-1],#the last column
                test_array[:,:-1],#all the columns except the last one
                test_array[:,-1]#the last column
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False)
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            #to get the best model score from the dict
            ##to get the best model name from the dict
            best_model_score=max(sorted(model_report.values()))#to get the best model score from the dict
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]#to get the best model name from the dict
            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=models[best_model_name]#to get the best model object from the dict
            )
            predicted= models[best_model_name].predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)



