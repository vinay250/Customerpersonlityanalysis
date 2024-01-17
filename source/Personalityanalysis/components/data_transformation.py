# data_transformation.py
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass
from joblib import dump  # Added import for dump
import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def create_new_feature(self, data):
        # Drop the 'Dt_Customer' column
        data = data.drop('Dt_Customer', axis=1)
        # Example: Create a new feature 'TotalSpend' as the sum of spending on different product categories
        data['TotalSpend'] = data[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts']].sum(axis=1)
        return data

    def train_evaluate_model(self, data):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(data.drop('Response', axis=1), data['Response'], test_size=0.2, random_state=42)

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

    def data_transformation(self, data):
        try:
            # Specify numerical and categorical features
            numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = data.select_dtypes(include=['object']).columns

            # Create transformers for numerical data
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Check if there are categorical features
            if not categorical_features.empty:
                # Create transformers for categorical data
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])

                # Create column transformer
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numerical_transformer, numerical_features),
                        ('cat', categorical_transformer, categorical_features)
                    ])
            else:
                # If no categorical features, use only numerical transformers
                preprocessor = numerical_transformer

            # Apply transformations
            transformed_data = preprocessor.fit_transform(data)

            # Save preprocessor object
            dump(preprocessor, self.data_transformation_config.preprocessor_obj_file_path)

            # Add logging statement
            logging.info("Preprocessing pickle file saved")

            return transformed_data

        except Exception as e:
            print("Exception occurred in the data_transformation:", str(e))
            raise e

    def run_data_transformation(self, data_path):
        try:
            # Load data
            data = self.load_data(data_path)

            # Data preprocessing
            data = self.create_new_feature(data)
            transformed_data = self.data_transformation(data)

            # Convert transformed_data to a Pandas DataFrame
            transformed_df = pd.DataFrame(transformed_data, columns=data.columns)

            # Display basic statistics, visualize, and train/evaluate the model
            print(transformed_df.describe())
            # Visualizations...
            self.train_evaluate_model(data)

        except Exception as e:
            print("Exception occurred in the run_data_transformation:", str(e))
            raise e

if __name__ == "__main__":
    # Specify the path to your data file
    data_path = 'E:/CustomerPersonalityAnalysis/artifacts/train.csv'

    # Initialize DataTransformation
    data_transformation_instance = DataTransformation()

    # Run data transformation
    data_transformation_instance.run_data_transformation(data_path)
