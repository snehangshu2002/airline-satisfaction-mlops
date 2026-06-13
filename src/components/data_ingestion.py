import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger_config import get_logger
from src.utils import load_params

logger = get_logger(os.path.splitext(os.path.basename(__file__))[0])




def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a csv file"""

    try:
        logger.info("Statring data ingestion")
        df = pd.read_csv(data_url)
        logger.debug(f"Loaded {df.shape[0]} rows")
        return df

    except Exception:
        logger.exception("Failed to load data from source")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprosess the data"""
    try:
        params = load_params()
        logger.info("Preprocessing stated")
        drop_cols = params["features"]["drop_cols"]

        df["Flight Distance"] = pd.to_numeric(df["Flight Distance"], errors="coerce")
        logger.debug("Convert Flight Distance column to numeric")
        df = df.drop(columns=drop_cols, errors="ignore")
        logger.debug(f"Dropped columns: {drop_cols}")

        df = df.dropna()
        logger.debug(f"Dropped nulls, remaining rows: {df.shape[0]}")

        logger.debug("Replaced 0s with median in rating columns")
        return df

    except Exception:
        logger.exception("Data preprocessing failed")
        raise


def save_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> None:
    """save the train and test datasets"""
    try:
        params = load_params()
        logger.info("Saving train/test splits")

        train_path = params["data"]["train_path"]
        test_path = params["data"]["test_path"]

        os.makedirs(os.path.dirname(train_path), exist_ok=True)

        train_data.to_csv(train_path, index=False)
        logger.debug(f"Train data saved to {train_path} — shape: {train_data.shape}")

        test_data.to_csv(test_path, index=False)
        logger.debug(f"Test data saved to {test_path} — shape: {test_data.shape}")

    except Exception:
        logger.exception("Failed to save train/test datasets")
        raise


def main():
    try:
        params = load_params()
        test_size = params["data"]["test_size"]
        raw_path = params["data"]["raw_path"]
        random_state = params["data"]["random_state"]

        df = load_data(data_url=raw_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=random_state
        )
        logger.info(
            f"Train/test split done — train: {train_data.shape}, test: {test_data.shape}"
        )

        save_data(train_data, test_data)
        logger.info("Data ingestion completed successfully")

    except Exception:
        logger.exception("Data ingestion pipeline failed")
        raise


if __name__ == "__main__":
    main()
