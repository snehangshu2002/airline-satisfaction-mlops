import yaml
from src.logger_config import get_logger
from src.exception import CustomException

from dotenv import load_dotenv
import os
import mlflow
load_dotenv()
import sys

logger = get_logger("Utils")
def load_params(path:str="params.yaml")->dict:
    """Load params.yaml and return as dict"""
    
    try:
        with open(path,"r") as f:
            params = yaml.safe_load(f)
        logger.debug(f"Params loaded from {path}")
        return params
    except Exception as e:
        CustomException(e,sys)

def setup_mlflow(experiment_name: str) -> None:
    """Configure MLflow to track on DagsHub."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    username     = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password     = os.environ.get("MLFLOW_TRACKING_PASSWORD")

    if not tracking_uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI not set in .env")
    if not username or not password:
        raise EnvironmentError(
            "MLFLOW_TRACKING_USERNAME or MLFLOW_TRACKING_PASSWORD not set in .env"
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking → {tracking_uri} | experiment → {experiment_name}")