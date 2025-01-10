import sys,os 
from dataclasses import dataclass
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

@dataclass 
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self, train_array, test_array):
         try:
             logging.info('splitting dependent and independent variables from train and test data')
             X_train, y_train, X_test,y_test = (
                 train_array[:,:-1],
                 train_array[:,-1],
                 test_array[:,:-1],
                 test_array[:, -1]

             )
             ## Train multiple models 
             # Model Evaluation 
             models = {
             'LogisticRegression' : LogisticRegression(),
             'RandomForestClassifier' : RandomForestClassifier(),
             'SVC': SVC(kernel='linear', random_state=42),
             'KNN': KNeighborsClassifier(n_neighbors=5)
           } 

             model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models) 
             print(model_report)
             print('\n ==================================================================================')
             logging.info(f'Model Report: {model_report}')

             #to get best model score from dictionary 
             best_model_score= max(sorted(model_report.values()))

             best_model_name = list(model_report.keys())[
                 list(model_report.values()).index(best_model_score)
                  ]
             best_model = models[best_model_name]
            
             print(f'Best model found, Model named : {best_model_name},\n Accuracy score: {best_model_score}')
             print('\n======================================================================================')
             logging.info(f'Best model found, Model named : {best_model_name}, Accuracy score: {best_model_score}')

             save_object(file_path=self.model_trainer_config.trained_model_file_path,
                         obj=best_model)

         except Exception as e:
             logging.info("exception occured at model training component ")
             raise CustomException(e,sys)    
          