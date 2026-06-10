import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.logger_config import get_logger
from src.exception import CustomException
from src.utils import load_params

params = load_params()

logger = get_logger( os.path.splitext(os.path.basename(__file__))[0])



# def load_model(file_path:str):
#     """Load trained model from a file"""
#     try:
#         logger.info("Model loading Started")
#         with open (file_path,'rb') as file:
#             model = joblib.load(file)
#             logger.debug("Model loaded Successful")
            
#             return model
#     except Exception as e:
#         CustomException(e,sys)


# def load_data(file_path:str)->pd.DataFrame:
#     """
#     Load data from a CSV file.
    
#     :param file_path: Path to the CSV file
#     :return: Loaded DataFrame
#     """
#     try:
#         logger.info("Load the data")
#         df=pd.read_csv(file_path)
#         logger.debug(f"data loaded from {file_path} with shape {df.shape}")
#         return df
#     except Exception as e:
#         raise CustomException(e,sys)

def evaluate_model(model,X_test:np.ndarray,y_test:np.ndarray)->dict:
     """Evaluate the model and return the evaluation metrics."""
     try:
         y_pred = model.predict(X_test)
         y_Pred_prob = model.predict_proba(X_test)[:,1]
         
         accuracy = accuracy_score(y_test,y_pred)
         logger.info(" test accuracy calculated")
         precision =precision_score(y_test,y_pred)
         logger.info(" test precision calculated")
         recall = recall_score(y_test,y_pred)
         logger.info(" test recall calculated")
         auc = roc_auc_score(y_test,y_Pred_prob)
         logger.info(" test auc calculated")
         
         metrics_dict = {
             "accuracy":accuracy,
             "precision":precision,
             "recall":recall,
             "auc":auc
         }
         
         logger.debug('Model evaluation metrics calculated')
         return metrics_dict
     
     except Exception as e:
         CustomException(e,sys)

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        CustomException(e,sys)
        
def main():
    try:
        
        categorical_cols = params["features"]["categorical_cols"]
        rating_cols = params["features"]["rating_cols"]
        target_col = params["features"]["target_col"]
        
        encoder = joblib.load(params["artifacts"]["encoder_path"])
        logger.debug("Encoder saved object loaded")
        scaler = joblib.load(params["artifacts"]["scaler_path"])
        logger.debug("Scaler  saved object loaded")
        label_encoder = joblib.load(params["artifacts"]["label_encoder_path"])
        logger.debug("Label_encoder saved object loaded")
        
        rating_medians = joblib.load(params["artifacts"]["rating_medians_path"])

        model = joblib.load(params["artifacts"]["model_path"])
        
        logger.info("All artifacts loaded")
        
        test_data = pd.read_csv(params["data"]["processed_test_path"])
        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]
        
        for col in rating_cols:
            if X_test[col].iloc[0]==0:
                X_test[col]=rating_medians[col]
        encoded = encoder.transform(X_test[categorical_cols])
        encoded_X_test = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(),
            dtype=int
        )
        # Drop categorical cols  and concat enocode  
        X_test=X_test.drop(columns=categorical_cols)
        X_test = pd.concat([X_test,encoded_X_test],axis=1)
        
        X_test = scaler.transform(X_test)
        
        metrics = evaluate_model(model,X_test,y_test)
        save_metrics(metrics,params["evaluate"]["metrics_path"])
        logger.info(f"Evaluation complete: {metrics}")
    except Exception as e:
        CustomException(e,sys)
        
if __name__ == "__main__":
    try:
        main()
    except CustomException:
        sys.exit(1)