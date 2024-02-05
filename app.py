from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and feature names
model = joblib.load('E:/CustomerPersonalityAnalysis/artifacts/trained_model.pkl')
feature_names = joblib.load('E:/CustomerPersonalityAnalysis/artifacts/feature_names.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get form data
        data = request.get_json()

        # Create a DataFrame with the user input
        user_data = pd.DataFrame(data, index=[0])

        # Ensure the DataFrame has the correct columns
        user_data = user_data.reindex(columns=feature_names, fill_value=0)

        # Use the trained model for prediction
        predictions = model.predict(user_data)

        # Map predictions to human-readable labels (adjust as needed)
        prediction_labels = {0: 'Not Interested', 1: 'Interested'}

        # Get the prediction for the first (and only) input
        prediction = predictions[0]

        # Format the response
        result = {
            "result": f"Analysis Results for {data['customerName']} - Product Feedback: {data['productFeedback']}, Customer Actions: {data['customerActions']}",
            "prediction": prediction_labels[prediction]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
