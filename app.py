import os
import sys
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Fetch absolute base path to locate models accurately on Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

vectorizer = None
model = None

# Safely load Vectorizer and Model
try:
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        print("--> Model and Vectorizer loaded successfully!", file=sys.stderr)
    else:
        print(f"--> ERROR: Pickle files not found in directory: {BASE_DIR}", file=sys.stderr)
except Exception as e:
    print(f"--> ERROR loading model or vectorizer: {e}", file=sys.stderr)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Return a clear message if files failed to load instead of 500
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model or Vectorizer file is missing on the server. Make sure model.pkl and vectorizer.pkl are uploaded to Git.'
        }), 500

    try:
        data = request.get_json(silent=True) or request.form
        text_input = data.get('text', '').strip()

        if not text_input:
            return jsonify({'error': 'Please enter text for sentiment analysis.'}), 400

        # Transform and predict
        transformed_input = vectorizer.transform([text_input])
        prediction = model.predict(transformed_input)[0]

        # Calculate confidence if supported
        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed_input)[0]
            confidence = round(float(max(probabilities)) * 100, 2)

        return jsonify({
            'text': text_input,
            'sentiment': str(prediction).capitalize(),
            'confidence': confidence
        })

    except Exception as e:
        print(f"--> Prediction Error: {e}", file=sys.stderr)
        return jsonify({'error': f"Internal error during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
