# scripts/api.py

from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# Path to the models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../models')

# Load the models
models = {
    "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, 'Logistic Regression.joblib')),
    "Decision Tree": joblib.load(os.path.join(MODELS_DIR, 'Decision Tree.joblib')),
    "Random Forest": joblib.load(os.path.join(MODELS_DIR, 'Random Forest.joblib')),
    "Gradient Boosting": joblib.load(os.path.join(MODELS_DIR, 'Gradient Boosting.joblib')),
}

@app.route('/')
def home():
    return "Welcome to the Credit Scoring Model API. Use the '/predict' endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON request data
        data = request.json
        
        # Convert input data to numpy array (you may need to adjust this based on your feature input)
        features = np.array([data['Recency'], data['Frequency'], data['Monetary'], data['Seasonality']]).reshape(1, -1)

        # Predict using a specific model, e.g., Logistic Regression
        prediction = models["Logistic Regression"].predict(features)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

