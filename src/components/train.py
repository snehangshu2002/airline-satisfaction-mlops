import pandas as pd
import numpy as np
from xgboost import XGBClassifier, train
import os
import sys
import joblib
from src.logger_config import get_logger
from src.exception import CustomException

logger = get_logger( os.path.splitext(os.path.basename(__file__))[0])

def load_data(file_path:str)->pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        logger.info("Load the data")
        df=pd.read_csv(file_path)
        logger.debug(f"data loaded from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e,sys)
        
def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> XGBClassifier:
    """
    Train the XGBoost model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained XGBoost model
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        logger.info('Initializing XGboot model with parameters: %s', params)
        
        model = XGBClassifier(**params)
        logger.debug('Model training started with %d samples', X_train.shape[0])
        model.fit(X_train, y_train)
        logger.debug('Model training completed')
        return model
    except Exception as e:
        raise CustomException(e,sys)
    
def save_model(model,file_path:str)->None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file:
            joblib.dump(model,file)
        logger.debug('Model saved to %s', file_path)
        
    except Exception as e:
        CustomException(e,sys)

    
def main():
    try:
        params ={
                    "colsample_bytree": 0.8,
                    "learning_rate": 0.1,
                    "max_depth": 7,
                    "n_estimators": 300,
                    "enable_categorical": False}
        
        train_data = load_data("data\processed\train.csv")
        X_train = train_data.drop(columns=['Satisfaction'])
        y_train = train_data['Satisfaction']
        
        model = train_model(X_train,y_train,params)
        
        model_save_path = "models/xgboost.pkl"
        save_model(model,model_save_path)
    except Exception as e:
        CustomException(e,sys)

if __name__ =="__main__":
    main()