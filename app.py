from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get form data
        data = request.get_json()

        # Extract product feedback and customer actions
        product_feedback = data.get('productFeedback', '')
        customer_actions = data.get('customerActions', '')

        # Simple logic to determine interest
        if 'love' in product_feedback.lower() or 'impressed' in product_feedback.lower():
            prediction = 'Interested'
        else:
            prediction = 'Not Interested'

        # Format the response
        result = {
            "result": f"Analysis Results for {data['customerName']} - Product Feedback: {product_feedback}, Customer Actions: {customer_actions}",
            "prediction": prediction
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
