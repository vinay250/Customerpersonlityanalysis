# prediction_pipeline.py
import pandas as pd
import joblib
from source.Personalityanalysis.components.data_transformation import DataTransformation

def run_prediction_pipeline(data_path):
    try:
        # Initialize DataTransformation
        data_transformation_instance = DataTransformation()

        # Run data transformation
        transformed_data = data_transformation_instance.run_data_transformation(data_path)

        # Load the trained model
        model_path = 'E:/CustomerPersonalityAnalysis/artifacts/trained_model.pkl'
        model = joblib.load(model_path)

        # Load the feature names
        feature_names_path = 'E:/CustomerPersonalityAnalysis/artifacts/feature_names.pkl'
        feature_names = joblib.load(feature_names_path)

        # Ensure feature_names is a list
        if not isinstance(feature_names, list):
            raise ValueError("Feature names should be a list.")

        # Set feature names for the model
        model.feature_names = feature_names

        #Perform prediction
        predictions = model.predict(transformed_data)

        # Output predictions (adjust this based on your needs)
        print("Predictions:", predictions)

    except Exception as e:
        print(f"Exception occurred in the run_prediction_pipeline: {str(e)}")

if __name__ == "__main__":
    # Specify the path to your data file (update this to the correct file path)
    data_path = 'E:/CustomerPersonalityAnalysis/artifacts/test.csv'

    # Run the prediction pipeline
    run_prediction_pipeline(data_path)
