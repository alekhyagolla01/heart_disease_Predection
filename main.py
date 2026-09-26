'''  Heart Disease using Machine Learning
'''

import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")
from log_code import setup_logging
logger = setup_logging("main") #cll the fun
from sklearn.model_selection import train_test_split
from yeo_timing import variable_transformation_outliers
from fs import select_best_columns
from all_models import common

class HEARTPREDECTION:
    def __init__(self,path):
        try:
            self.path=path
            self.df=pd.read_csv(self.path)
            logger.info(f"The Number of rows and columns was :{self.df.shape}")
            logger.info(f"Null values in the values: {self.df.isnull().sum()}")
            self.X = self.df.iloc[:, :-1]  # all indepent columns
            self.y = self.df.iloc[:, -1]  # all dependent columns

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            logger.info(f"Training dataset size : \n {self.X_train.shape}  => {self.y_train.shape}")
            logger.info(f"Testing dataset size : \n {self.X_test.shape} => {self.y_test.shape}")


        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def vt_outliers(self):
        try:
            self.X_train_numerical, self.X_test_numerical = variable_transformation_outliers( self.X_train,self.X_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : "f"{er_type} : reason : {er_msg}")

    def feature_selection(self):
        try:
            self.X_train_numerical, self.X_test_numerical = select_best_columns(self.X_train_numerical,self.X_test_numerical,self.y_train,self.y_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def spliting_data_clearlly(self):
        try:
            numerical_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                                 'oldpeak', 'slope', 'ca', 'thal']

            self.X_train_numerical = self.X_train[numerical_columns]
            self.X_test_numerical = self.X_test[numerical_columns]

            logger.info(f"Numerical X_train columns : {self.X_train_numerical.columns}")
            logger.info(f"Numerical X_test columns : {self.X_test_numerical.columns}")

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")

    def cat_to_numerical(self):
        try:
            self.final_training_data = \
                self.X_train_numerical.copy()
            self.final_testing_data = \
                self.X_test_numerical.copy()
            logger.info(
                f"Final Training Data : "
                f"{self.final_training_data.shape}"
            )
            logger.info(
                f"Final Testing Data : "
                f"{self.final_testing_data.shape}"
            )
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} "f": due to : {er_type} : reason : {er_msg}")

    def train_all_models(self):
        try:
            common(
                self.final_training_data,
                self.y_train,
                self.final_testing_data,
                self.y_test
            )

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(
                f"Error in line no : {er_line.tb_lineno} : "
                f"due to : {er_type} : reason : {er_msg}"
            )
if __name__ == "__main__":
    try:
        obj = HEARTPREDECTION("heart.csv")
        obj.spliting_data_clearlly()
        obj.vt_outliers()
        obj.feature_selection()
        obj.cat_to_numerical()
        obj.train_all_models()

    except Exception as e:
        er_type,er_msg, er_line=sys.exc_info()
        logger.info(f"Error in line no: {er_line.tb_lineno} : due to : {er_type} : reason :{er_msg}")