import pandas as pd
import numpy as np
from xgboost import XGBClassifier, train
import os
import sys
import joblib
from src.logger_config import get_logger
from src.exception import CustomException
from src.utils import load_params
# from dvclive import Live
import mlflow
import mlflow.xgboost

logger = get_logger( os.path.splitext(os.path.basename(__file__))[0])

params = load_params()

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
        model_params = {
            "n_estimators":     params["model"]["n_estimators"],
            "learning_rate":    params["model"]["learning_rate"],
            "max_depth":        params["model"]["max_depth"],
            "colsample_bytree": params["model"]["colsample_bytree"],
            "subsample":        params["model"]["subsample"],
            "random_state":     params["model"]["random_state"],
            "enable_categorical": params["model"]["enable_categorical"],
            "eval_metric": "logloss"
        }
        
        train_data = load_data(params["data"]["processed_train_path"])
        target = params["features"]["target_col"]
        
        X_train = train_data.drop(columns=[target])
        y_train = train_data[target]
        
        # Resume the MLflow run started in train.py
        run_id_path = "reports/mlflow_run_id.txt"
        if os.path.exists(run_id_path):
            with open(run_id_path) as f:
                run_id = f.read().strip()
            logger.info(f"Resuming MLflow run: {run_id}")
        else:
            run_id = None
            logger.warning("No MLflow run ID found — starting a new run")
        
        with mlflow.start_run(run_id=run_id) as run:
            logger.info(f"MLflow run started: {run.info.run_id}")
            
            #Log all hyperparameter
            
            mlflow.log_params(model_params)
            mlflow.log_param("train_samples",X_train.shape[0])
            mlflow.log_param("n_features",X_train.shape[1])
            
            model = train_model(X_train,y_train,model_params)
            
            mlflow.xgboost.log_model(
                    model,
                    name="xgboost-model",
                    input_example=X_train.iloc[:5]   
                )

            
             
            model_save_path = params["artifacts"]["model_path"]
            save_model(model,model_save_path)
            # mlflow.log_artifact(model_save_path,artifact_path="model")
            
            

            
            
            logger.info("Training pipeline completed successfully")
    except Exception as e:
        CustomException(e,sys)
        
        
if __name__ == "__main__":
    try:
        main()
    except CustomException:
        sys.exit(1)
