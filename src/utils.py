import yaml
from src.logger_config import get_logger
from src.exception import CustomException
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
        