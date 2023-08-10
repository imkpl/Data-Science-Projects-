import sys
from dataclasses import dataclass #Importing dataclasses
import os

#Now import all the required libraries 
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class DataTransformationConfig:  #Now create similar class used in data_ingestion
    preprocesser_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

    
class DataTransformation:

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self): #Use for the transforamtion of the data
         try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(  #creating the pipeline for numerical variable tansformation
                steps=[
                ("imputer",SimpleImputer(strategy="median")), #to handle missing values 
                ("scaler",StandardScaler()) #for normalization of the dataset

                ]
            )

            cat_pipeline = Pipeline (
                steps= [ #using each of the techniques for categorical variable transformation
                ("imputer", SimpleImputer (strategy="most_frequent")), #To handle missing value
                ("one_hot_encoder", OneHotEncoder()), #To create dummy variables
                ("Scaler", StandardScaler(with_mean= False)) 
                ]


            )

            logging.info('Transformation of Categorical and Numerical features is successfully completed')

            preprocessor = ColumnTransformer( #Creating new variable preprocessor to create a unified pipeline for cat and num features 

                [
                ("num_pipeline",num_pipeline,numerical_columns), # for integrate num pipline using name of pipline and giving num coloumn names 
                ("cat_pipelines",cat_pipeline,categorical_columns) # similarly for cat using name of pipline and giving cat coloumn names 
                ]
            )

            return preprocessor #now return the preprocessor variable containing all the transforamtion of the cat and num features 
         
         except Exception as e:
            raise CustomException(e,sys) #To raise exception if something goes wrong
    
    
    def initiate_data_transformation(self,train_path,test_path):#To actually initial transformation of the training and testing dataset
        try:
            train_df=pd.read_csv(train_path) #to read file from the training file path given as input of the initiate_data_transformation fuction 
            test_df=pd.read_csv(test_path)

            logging.info("Training and testing dataset has been read")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object() #To call the preproccer object created earlier in get_data_transformer_object function

            target_column_name="math_score" #output variable
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object( #To save this pkl file in same location we also added this into utils file and imported save_object from utils earlier 

                file_path=self.data_transformation_config.preprocesser_obj_file_path, #file save file path 
                obj=preprocessing_obj #to save object

            ) 

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)
            





            
             


        
        