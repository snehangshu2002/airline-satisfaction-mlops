import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from typing import Tuple

from src.logger_config import get_logger
from src.exception import CustomException

# Configure logger
logger = get_logger("Feature_Engineering")

rating_cols = [
    'Departure and Arrival Time Convenience', 'Ease of Online Booking',
    'Check-in Service', 'Online Boarding', 'Gate Location',
    'On-board Service', 'Seat Comfort', 'Leg Room Service',
    'Cleanliness', 'Food and Drink', 'In-flight Service',
    'In-flight Wifi Service', 'In-flight Entertainment', 'Baggage Handling'
]

def feature_engineering(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, OneHotEncoder, StandardScaler, LabelEncoder]:
    """Apply encoder and scaler to the data."""
    try:
        logger.info("Starting feature engineering process")
        categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

        # Initialize OneHotEncoder
        # Note: Specifying drop='first' is not compatible with handle_unknown='ignore'.
        # Since we use tree-based models like XGBoost, drop='first' is not required.
        logger.info("Initializing OneHotEncoder and fitting on training categorical data")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype="i2")
        
        X_train = train_data.drop(columns=['Satisfaction'])
        y_train = train_data['Satisfaction']  
        
        X_test = test_data.drop(columns=['Satisfaction'])
        y_test = test_data['Satisfaction']    
        
        # Fit and transform the categorical columns on training set
        encoded_features = encoder.fit_transform(X_train[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_features, 
            columns=encoder.get_feature_names_out(),
            index=X_train.index,
            dtype=int
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
            dtype=int
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
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        logger.info("Feature engineering process completed successfully")
        return X_train, y_train, X_test, y_test, encoder, scaler, label_encoder

    except Exception as e:
        raise CustomException(e, sys)

def main():
    try:
        # Define paths
        raw_train_path = "data/raw/train.csv"
        raw_test_path = "data/raw/test.csv"
        processed_dir = "data/processed"
        models_dir = "models"
        
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info(f"Loading raw train data from {raw_train_path}")
        train_df = pd.read_csv(raw_train_path)
        logger.info(f"Loading raw test data from {raw_test_path}")
        test_df = pd.read_csv(raw_test_path)
        
        # Apply feature engineering
        X_train, y_train, X_test, y_test, encoder, scaler, label_encoder = feature_engineering(train_df, test_df)
        
        # Combine X and y back to DataFrames for saving
        train_processed = X_train.copy()
        train_processed['Satisfaction'] = y_train
        
        test_processed = X_test.copy()
        test_processed['Satisfaction'] = y_test
        
        # Save processed data
        train_processed_path = os.path.join(processed_dir, "train.csv")
        test_processed_path = os.path.join(processed_dir, "test.csv")
        
        logger.info(f"Saving processed train data to {train_processed_path}")
        train_processed.to_csv(train_processed_path, index=False)
        
        logger.info(f"Saving processed test data to {test_processed_path}")
        test_processed.to_csv(test_processed_path, index=False)
        
        # Save preprocessors
        logger.info("Saving preprocessing models/objects to models directory")
        joblib.dump(encoder, os.path.join(models_dir, "encoder.joblib"))
        joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
        joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.joblib"))
        
        logger.info("Feature engineering script executed successfully")
        
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        main()
    except CustomException:
        sys.exit(1)