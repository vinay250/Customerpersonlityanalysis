import os
import sys
from datetime import datetime
from source.Personalityanalysis.logger import logging
from source.Personalityanalysis.exception import customexception
import pandas as pd
from source.Personalityanalysis.components.data_transformation import DataTransformation
from source.Personalityanalysis.components.model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

def setup_logging():
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    log_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_path, exist_ok=True)
    LOG_FILEPATH = os.path.join(log_path, LOG_FILE)

    logging.basicConfig(
        level=logging.INFO,
        filename=LOG_FILEPATH,
        format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
    )

class DataIngestion:
    def __init__(self):
        setup_logging()
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('E:/CustomerPersonalityAnalysis/notebooks/preprocessed_data.csv')
            logging.info('Read the dataset as a dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise customexception(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Specify the path to your data file
    data_path = 'E:/CustomerPersonalityAnalysis/artifacts/train.csv'

    # Initialize DataTransformation
    data_transformation_instance = DataTransformation()

    # Run data transformation
    data_transformation_instance.run_data_transformation(data_path)

    # Initialize ModelTrainer
    model_trainer = ModelTrainer(data_transformation_instance)

    # Run the model trainer
    model_trainer.run_model_trainer(data_path)
