import os
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the TF-IDF Vectorizer and Model pickles
# Replace 'vectorizer.pkl' and 'model.pkl' with your exact pickle file names
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    print("Model and Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    vectorizer, model = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model or Vectorizer is not loaded properly.'}), 500

    try:
        data = request.get_json(silent=True) or request.form
        text_input = data.get('text', '').strip()

        if not text_input:
            return jsonify({'error': 'Please provide text for sentiment analysis.'}), 400

        # Transform input text using vectorizer
        transformed_input = vectorizer.transform([text_input])

        # Predict sentiment label
        prediction = model.predict(transformed_input)[0]

        # Predict probability if model supports it
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed_input)[0]
            confidence = round(float(max(probabilities)) * 100, 2)

        # Standardize prediction text for UI output
        sentiment = str(prediction).capitalize()
        
        return jsonify({
            'text': text_input,
            'sentiment': sentiment,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
