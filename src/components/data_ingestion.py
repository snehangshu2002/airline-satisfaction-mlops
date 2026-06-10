import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

from src.logger_config import get_logger
from src.exception import CustomException

logger = get_logger( os.path.splitext(os.path.basename(__file__))[0])

rating_cols = [
    'Departure and Arrival Time Convenience', 'Ease of Online Booking',
    'Check-in Service', 'Online Boarding', 'Gate Location',
    'On-board Service', 'Seat Comfort', 'Leg Room Service',
    'Cleanliness', 'Food and Drink', 'In-flight Service',
    'In-flight Wifi Service', 'In-flight Entertainment', 'Baggage Handling'
]

def load_data(data_url:str)->pd.DataFrame:
    """Load data from a csv file"""
    
    try:
        logger.info("Statring data ingestion")
        df = pd.read_csv(data_url)
        logger.debug(f"Loaded {df.shape[0]} rows")
        return df
    
    except Exception as e:
        raise CustomException(e, sys)

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """Preprosess the data"""
    try:
        logger.info("Preprocessing stated")
        df['Flight Distance'] = pd.to_numeric(df['Flight Distance'], errors='coerce')
        logger.debug("Convert the Flight Distance column to numeric")
        df = df.drop(columns=['ID'], errors='ignore')      
        logger.debug("Drop the ID column")    
        
        for col in rating_cols:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

        return df
        
    except Exception as e:
        raise CustomException(e, sys)

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    """save the train and test datasets"""
    try:
        logger.info("Data saving started")
        raw_data_path=os.path.join(data_path,"raw")
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        logger.debug(
                f"Train data saved in {os.path.join(raw_data_path, 'train.csv')}"
            )
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logger.debug(
                f"Test data saved in {os.path.join(raw_data_path, 'train.csv')}"
            )
    except Exception as e:
        raise CustomException(e, sys)
    
def main():
    try:
        test_size = 0.2
        
        data_url=input("Enter Your Data path : ")
        df = load_data(data_url=data_url)
        final_df = preprocess_data(df)
        train_data,test_data = train_test_split(final_df,test_size=test_size,random_state=42)
        
        save_data(train_data,test_data,data_path='./data')
        
    except Exception as e:
        raise CustomException(e, sys)
        
if __name__=="__main__":
    try:
        main()
    except CustomException:
        sys.exit(1)