import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Update this import
from data_transformation import DataTransformation




def run_model_trainer(data_path):
    try:
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

        # Model Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')

        # Classification Report
        print('Classification Report:')
        print(classification_report(y_test, y_pred))

        # Save the trained model
        model_path = 'E:/CustomerPersonalityAnalysis/artifacts/trained_model.pkl'
        joblib.dump(model, model_path)
        print(f'Trained model saved at: {model_path}')

    except Exception as e:
        print(f"Exception occurred in the run_model_trainer: {str(e)}")

if __name__ == "__main__":
    # Specify the path to your data file
    data_path = 'E:/CustomerPersonalityAnalysis/artifacts/train.csv'

    # Run the model trainer
    run_model_trainer(data_path)
