##willl have something which will have common for whole src section 
import sys,os 
import pandas as pd 
import numpy as np 
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score , precision_score , recall_score, classification_report,f1_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(models)):
            model=list(models.values())[i]

            #training the model 
            model.fit(X_train,y_train)

            # make prediction on testing data
            y_test_pred=model.predict(X_test)

            #get Accuracy score for train and test data 
            #train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info("Exception occured during model training ")
        raise CustomException(e,sys)

 
           
    
