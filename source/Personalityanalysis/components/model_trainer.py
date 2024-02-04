import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from dataclasses import dataclass
from source.Personalityanalysis.exception import CustomException
from source.Personalityanalysis.logger import logging

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'trained_model.pkl')

class ModelTrainer:
    def __init__(self, data_transformation_instance):
        self.model_trainer_config = ModelTrainerConfig()
        self.data_transformation_instance = data_transformation_instance

    def train_model(self, transformed_data):
        try:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                transformed_data.drop('Response', axis=1),
                transformed_data['Response'],
                test_size=0.2,
                random_state=42
            )

            # Model Training
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Model Evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy:.2f}')

            # Classification Report
            print('Classification Report:')
            print(classification_report(y_test, y_pred))

            # Save the trained model
            dump(model, self.model_trainer_config.model_path)
            print(f"Model saved to: {self.model_trainer_config.model_path}")

            logging.info("Trained model saved successfully.")

        except Exception as e:
            print(f"Exception during training: {str(e)}")
            raise CustomException(e, sys)

    def run_model_trainer(self, data_path):
        try:
            # Run data transformation
            transformed_data = self.data_transformation_instance.run_data_transformation(data_path)

            # Train the model using the transformed data
            self.train_model(transformed_data)

        except Exception as e:
            print("Exception occurred in the run_model_trainer:", str(e))
            raise e

if __name__ == "__main__":
    # Specify the path to your data file
    data_path = 'E:/CustomerPersonalityAnalysis/artifacts/train.csv'

    # Initialize DataTransformation
    data_transformation_instance = DataTransformation()

    # Initialize ModelTrainer
    model_trainer = ModelTrainer(data_transformation_instance)

    # Run the model trainer
    model_trainer.run_model_trainer(data_path)
