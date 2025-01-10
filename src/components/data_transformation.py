from sklearn.impute  import SimpleImputer ## Handling the Missing Values
from sklearn.preprocessing import StandardScaler #Handling Featue Scalling 
from sklearn.preprocessing import OrdinalEncoder #handling Encoding 
## pipeline 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os 
from dataclasses import dataclass
import pandas as pd 
import numpy as np 

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

#Data transformation config 
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


#data transformation class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformation_object(self,df):
        try:
            logging.info('Data Transformation initiated')
            #convert  malign to 1 and benign to 0 
            #df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
            #list of numerical columns 
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

            #Exclude the target column if present in the list
            target_column = 'diagnosis'
            numerical_cols = [col for col in numerical_cols if col != target_column]

            logging.info('Pipeline Initiated')

            #Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )


            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols)
            ])

            return preprocessor
            logging.info('Pipeline Completed')
        
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
    


    def initiate_data_transformation(self,train_path, test_path):
        try:
            #Reading  train and test data 
            train_df = pd.read_csv(train_path) 
            test_df = pd.read_csv(test_path) 

            logging.info('Read train and test data completed ')
            logging.info(f'Train Dataframe head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe head : \n{train_df.head().to_string()}')

            logging.info('obtaining preprocessing object')
            
            #doing the preprocesing and fe for the traing data 
            logging.info("starting preprocessing and FE for training data ")

            #preprocessing_obj = self.get_data_transformation_object(df=train_df) #object 
             
            train_df = train_df.drop('Unnamed: 32', axis=1) #drop the null value columns
            
            
            #list of column to drop if correlation value is greater than .92
            train_df['diagnosis'] = train_df['diagnosis'].map({'M': 1, 'B': 0})
            corr_matrix=train_df.corr().abs()
            mask=np.triu(np.ones_like(corr_matrix,dtype=bool))
            tr_df=corr_matrix.mask(mask)
            to_drop=[x for x in tr_df.columns if any(tr_df[x]>.92)]
            to_drop

            #dropping the columns along with id as its not required 
            train_df.drop(to_drop,axis=1,inplace=True)
            train_df.drop('id',axis=1,inplace=True) #this have 23 features 

            preprocessing_obj = self.get_data_transformation_object(df=train_df) #this have a list of 22 input feature which is num tpye

            #dividing the input and target feature of train dataset 

            input_feature_train_df = train_df.drop('diagnosis',axis=1)
            target_feature_train_df = train_df['diagnosis']
            logging.info("preprocessing and FE for training data is completed")

            #preprocessing and fe for the testing data 
            logging.info("starting preprocessing and FE for testing  data ")

            #preprocessing_obj = self.get_data_transformation_object(df=test_df) #object 
             
            test_df = test_df.drop('Unnamed: 32', axis=1) #drop the null value columns

            #list of column to drop if correlation value is greater than .92
            test_df['diagnosis'] = test_df['diagnosis'].map({'M': 1, 'B': 0})
            corr_matrix=test_df.corr().abs()
            mask=np.triu(np.ones_like(corr_matrix,dtype=bool))
            tr_df=corr_matrix.mask(mask)
            to_drop=[x for x in tr_df.columns if any(tr_df[x]>.92)]
            to_drop

            #dropping the columns along with id as its not required 
            test_df.drop(to_drop,axis=1,inplace=True)
            test_df.drop('id',axis=1,inplace=True) #this have 23 features

            preprocessing_obj = self.get_data_transformation_object(df=test_df) #this have list of 22 input features

            #dividing the input and target feature of train dataset 

            input_feature_test_df = test_df.drop('diagnosis',axis=1)
            target_feature_test_df = test_df['diagnosis']
            logging.info("preprocessing and FE for training data is completed")

            #apply the transformation 

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("applying preprocessing object on training and testing datasets")
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessing_obj
                        )
            
            logging.info("preprocessing pickle is created and saved ")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e,sys)

            

    