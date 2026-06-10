import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score,ConfusionMatrixDisplay,confusion_matrix

from src.logger_config import get_logger
from src.exception import CustomException
from src.utils import load_params
from dvclive import Live

params = load_params()

logger = get_logger( os.path.splitext(os.path.basename(__file__))[0])


def evaluate_model(model,X_test:np.ndarray,y_test:np.ndarray,live:Live)->dict:
     """Evaluate the model and return the evaluation metrics."""
     try:
         y_pred = model.predict(X_test)
         y_pred_prob = model.predict_proba(X_test)[:,1]
         
         accuracy = accuracy_score(y_test,y_pred)
         logger.info(" test accuracy calculated")
         precision =precision_score(y_test,y_pred)
         logger.info(" test precision calculated")
         recall = recall_score(y_test,y_pred)
         logger.info(" test recall calculated")
         auc = roc_auc_score(y_test,y_pred_prob)
         logger.info(" test auc calculated")
         
         metrics_dict = {
             "accuracy":accuracy,
             "precision":precision,
             "recall":recall,
             "auc":auc
         }
         
         for metric_name,value in metrics_dict.items():
             live.log_metric(metric_name,value)
             logger.debug(f"Logged {metric_name}: {value}")
             
        
         cm = confusion_matrix(y_test,y_pred)
         live.log_sklearn_plot(
             "confusion_matrix",
             y_test.tolist(),
             y_pred.tolist(),
             name="Confusion_matirx",
             normalized=True
         )
         logger.debug("Confusion matrix plot logged")
         
         # Log ROC curve
         # Log ROC curve
         live.log_sklearn_plot(
            "roc",
            y_test.tolist(),
            y_pred_prob.tolist(),
            name="roc_curve"
         )
         logger.debug("ROC curve plot logged")

         
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
        target_col = params["features"]["target_col"]
        model = joblib.load(params["artifacts"]["model_path"])
        logger.info("Model loaded")

        test_data = pd.read_csv(params["data"]["processed_test_path"])
        if target_col not in test_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in processed test data. Available columns: {list(test_data.columns)}")

        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]

        with Live(dir="dvclive",resume=True) as live:
            
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
