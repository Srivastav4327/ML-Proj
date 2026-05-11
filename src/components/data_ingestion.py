#reading data from different sources and loading it to the local file system
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:#inputs for data ingestion component
    train_data_path: str=os.path.join('artifacts','train.csv')#to save the train data in artifacts folder
    test_data_path: str=os.path.join('artifacts','test.csv')#to save the test data in artifacts folder
    raw_data_path: str=os.path.join('artifacts','data.csv')#to save the raw data in artifacts folder4

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()#creating an object of DataIngestionConfig class
    def initiate_data_ingestion(self):#function to read the data and split it into train and test data
        logging.info("Entered the data ingestion component")
        try:
            df=pd.read_csv('notebook/data/stud.csv')#reading the data from the csv file
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)#creating the directory if it does not exist
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)#saving the raw data to the local file system

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)#splitting the data into train and test data
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)#saving the train data to the local file system
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)#saving the test data to the local file system

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
     
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)