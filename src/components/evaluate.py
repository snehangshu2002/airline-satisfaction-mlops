import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score,ConfusionMatrixDisplay,confusion_matrix,RocCurveDisplay

from src.logger_config import get_logger
from src.exception import CustomException
from src.utils import load_params
# from dvclive import Live
import mlflow
import mlflow.xgboost

params = load_params()

logger = get_logger( os.path.splitext(os.path.basename(__file__))[0])


def evaluate_model(model,X_test:np.ndarray,y_test:np.ndarray)->dict:
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
                  
         logger.debug('Model evaluation metrics calculated')
         
         return metrics_dict,y_pred,y_pred_prob
     
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

def save_confusion_matrix_plot(y_test,y_pred,save_path:str)->str:
    
    try:
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        fig,ax=plt.subplot(figsize=(6,5))
        cm = confusion_matrix(y_test,y_pred)
        
        disp =ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Satisfied","Satisfied"]
        )
        
        disp.plot(ax=ax,colorbar=True,cmap="Blues")
        
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"Confusion matrix plot saved to {save_path}")
        
        return save_path
    except Exception as e:
        CustomException(e,sys)

def save_roc_curve_plot(model, X_test, y_test, save_path: str) -> str:
    """Generate and save ROC curve as PNG for MLflow."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title("ROC Curve")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"ROC curve plot saved to {save_path}")
        return save_path
    except Exception as e:
        raise CustomException(e, sys)


def save_feature_importance_plot(model, feature_names: list, save_path: str) -> str:
    """Generate and save top-20 feature importance plot for MLflow."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # top 20
 
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            [feature_names[i] for i in reversed(indices)],
            importances[list(reversed(indices))],
            color="steelblue"
        )
        ax.set_title("Top 20 Feature Importances")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"Feature importance plot saved to {save_path}")
        return save_path
    except Exception as e:
        raise CustomException(e, sys)
        
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
            
            metrics, y_pred, y_pred_prob = evaluate_model(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            
            save_metrics(metrics,params["evaluate"]["metrics_path"])
            
            mlflow.log_artifact(params["evaluate"]["metrics_path"])
            
            # Confusion matrix plot
              # Confusion matrix plot
            cm_plot_path = save_confusion_matrix_plot(
                y_test, y_pred,
                save_path="reports/plots/confusion_matrix.png"
            )
            mlflow.log_artifact(cm_plot_path,artifact_path="plots")
            
             # ROC curve plot
            roc_plot_path = save_roc_curve_plot(
                model, X_test, y_test,
                save_path="reports/plots/roc_curve.png"
            )
            mlflow.log_artifact(roc_plot_path, artifact_path="plots")
 
            # Feature importance plot
            fi_plot_path = save_feature_importance_plot(
                model,
                feature_names=list(X_test.columns),
                save_path="reports/plots/feature_importance.png"
            )
            mlflow.log_artifact(fi_plot_path, artifact_path="plots")
            logger.info(f"Evaluation complete: {metrics}")
    except Exception as e:
        CustomException(e,sys)
        
if __name__ == "__main__":
    try:
        main()
    except CustomException:
        sys.exit(1)
