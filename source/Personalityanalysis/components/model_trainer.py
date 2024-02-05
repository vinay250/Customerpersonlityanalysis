# model_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging  # Import the logging module
from source.Personalityanalysis.components.data_transformation import DataTransformation

def run_model_trainer(data_path):
    try:
        # Initialize logging
        logging.basicConfig(filename='E:/CustomerPersonalityAnalysis/logs/model_trainer.log',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize DataTransformation
        data_transformation_instance = DataTransformation()

        # Run data transformation
        transformed_data = data_transformation_instance.run_data_transformation(data_path)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            transformed_data.drop('Response', axis=1),
            transformed_data['Response'],
            test_size=0.2,
            random_state=42
        )

        # Ensure y_train and y_test are converted to integers
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # Model Training
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Save the trained model and feature names
        model_path = 'E:/CustomerPersonalityAnalysis/artifacts/trained_model.pkl'
        joblib.dump(model, model_path)

        # Save feature names
        feature_names_path = 'E:/CustomerPersonalityAnalysis/artifacts/feature_names.pkl'
        joblib.dump(list(X_train.columns), feature_names_path)

        # Log the information
        logging.info(f'Trained model saved at: {model_path}')
        logging.info(f'Feature names saved at: {feature_names_path}')

    except Exception as e:
        # Log the exception
        logging.exception(f"Exception occurred in the run_model_trainer: {str(e)}")

if __name__ == "__main__":
    # Specify the path to your data file
    data_path = 'E:/CustomerPersonalityAnalysis/artifacts/train.csv'

    # Run the model trainer
    run_model_trainer(data_path)
