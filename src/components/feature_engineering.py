import os
import sys
from typing import Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger_config import get_logger
from src.utils import load_params, setup_mlflow

params = load_params()

# Configure logger
logger = get_logger(os.path.splitext(os.path.basename(__file__))[0])


def feature_engineering(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    OneHotEncoder,
    StandardScaler,
    LabelEncoder,
]:
    """Apply encoder and scaler to the data."""
    try:
        logger.info("Starting feature engineering process")
        categorical_cols = params["features"]["categorical_cols"]
        rating_cols = params["features"]["rating_cols"]
        target_col = params["features"]["target_col"]

        logger.info(
            "Initializing OneHotEncoder and fitting on training categorical data"
        )
        encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", dtype="i2"
        )

        X_train = train_data.drop(columns=target_col)
        y_train = train_data[target_col]

        X_test = test_data.drop(columns=target_col)
        y_test = test_data[target_col]

        # Fit and transform the categorical columns on training set
        logger.info("Fitting OneHotEncoder on train categorical columns")
        encoded_features = encoder.fit_transform(X_train[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_features,
            columns=encoder.get_feature_names_out(),
            index=X_train.index,
            dtype=int,
        )

        X_train = X_train.drop(columns=categorical_cols)
        X_train = pd.concat([X_train, encoded_df], axis=1)

        # Transform categorical columns on test set
        logger.info("Transforming test categorical data")
        encoded_test = encoder.transform(X_test[categorical_cols])
        encoded_df_test = pd.DataFrame(
            encoded_test,
            columns=encoder.get_feature_names_out(),
            index=X_test.index,
            dtype=int,
        )
        X_test = X_test.drop(columns=categorical_cols)
        X_test = pd.concat([X_test, encoded_df_test], axis=1)

        # Encode Target (Satisfaction)
        logger.info("Applying LabelEncoder to targets")
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # Scale Features
        logger.info("Applying StandardScaler to features")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert scaled arrays back to DataFrames to preserve column names and indices
        X_train = pd.DataFrame(
            X_train_scaled, columns=X_train.columns, index=X_train.index
        )
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        rating_medians = {}

        for col in rating_cols:
            temp = X_train[col].replace(0, np.nan)
            rating_medians[col] = temp.median()

        logger.info("Feature engineering process completed successfully")
        return (
            X_train,
            y_train,
            X_test,
            y_test,
            encoder,
            scaler,
            label_encoder,
            rating_medians,
        )

    except Exception as e:
        raise CustomException(e, sys)


def main():
    try:
        setup_mlflow("airline-satisfaction_3")
        # Define paths
        raw_train = params["data"]["train_path"]
        raw_test = params["data"]["test_path"]
        processed_train = params["data"]["processed_train_path"]
        processed_test = params["data"]["processed_test_path"]
        models_dir = params["artifacts"]["models_dir"]

        os.makedirs(os.path.dirname(processed_train), exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        logger.info(f"Loading raw train data from {raw_train}")
        train_df = pd.read_csv(raw_train)
        logger.info(f"Loading raw test data from {raw_test}")
        test_df = pd.read_csv(raw_test)

        # Apply feature engineering
        (
            X_train,
            y_train,
            X_test,
            y_test,
            encoder,
            scaler,
            label_encoder,
            rating_medians,
        ) = feature_engineering(train_df, test_df)

        # Combine X and y back to DataFrames for saving
        train_processed = X_train.copy()
        train_processed[params["features"]["target_col"]] = y_train

        test_processed = X_test.copy()
        test_processed[params["features"]["target_col"]] = y_test

        # Save processed data
        train_processed_path = os.path.join(processed_train)
        test_processed_path = os.path.join(processed_test)

        logger.info(f"Saving processed train data to {train_processed_path}")
        train_processed.to_csv(train_processed_path, index=False)

        logger.info(f"Saving processed test data to {test_processed_path}")
        test_processed.to_csv(test_processed_path, index=False)

        # Save preprocessors
        logger.info("Saving preprocessing models/objects to models directory")
        joblib.dump(encoder, params["artifacts"]["encoder_path"])
        joblib.dump(scaler, params["artifacts"]["scaler_path"])
        joblib.dump(label_encoder, params["artifacts"]["label_encoder_path"])
        joblib.dump(rating_medians, params["artifacts"]["rating_medians_path"])
        mlflow.set_experiment("airline-satisfaction_2")
        with mlflow.start_run() as run:
            mlflow.log_artifact(
                params["artifacts"]["encoder_path"], artifact_path="preprocessors"
            )
            mlflow.log_artifact(
                params["artifacts"]["scaler_path"], artifact_path="preprocessors"
            )
            mlflow.log_artifact(
                params["artifacts"]["label_encoder_path"], artifact_path="preprocessors"
            )
            mlflow.log_artifact(
                params["artifacts"]["rating_medians_path"],
                artifact_path="preprocessors",
            )
            logger.info("Preprocessor artifacts logged to MLflow")

            os.makedirs("reports", exist_ok=True)
            with open("reports/mlflow_run_id.txt", "w") as f:
                f.write(run.info.run_id)
            logger.info(f"Run ID saved : {run.info.run_id}")

        logger.info("Feature engineering script executed successfully")

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        main()
    except CustomException:
        sys.exit(1)
