import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn .model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig: #This is to get dataset and use default dataclass and 
    #use DataIngestionConfig class since here we are only defining variable and don't need init functions 
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


#Now we are making a new class for ingenstion of the data
class DataIngestion:
    def __init__(self):
        #Lets create new varibale ingestion_config based on DataIngestionConfig class
        self.ingestion_config=DataIngestionConfig() #Instance of the previously discussed DataIngestionConfig class is 
        #created and assigned to the attribute ingestion_config. This attribute will store the paths for the raw, 
        # train, and test data files as kind of subvariable.

    def initiate_data_ingestion(self): #To manage process of reading a CSV file, splitting the data, and saving the subsets.
        # Logging the entry into the data ingestion method
        logging.info("Entered the data ingestion method or component")
        try:
            # Reading the CSV file into a DataFrame using relative path
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as a DataFrame')#use to indefity exactly where error has occured

            # Creating the directory for the training data path if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Saving the entire DataFrame to the raw data path index = false to remove row or index and header is true to save columns names
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Splitting the DataFrame into training and test sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) 

            # Saving the training set to the train data path
            train_set.to_csv(self.ingestion_config.train_data_path, index=False , header=True)

            # Saving the test set to the test data path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Returning the paths to the training and test data files
            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raising a custom exception if any error occurs
            raise CustomException(e, sys)
       
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()


