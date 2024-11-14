from flask import Flask, request, jsonify, Blueprint
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Blueprint('app2', __name__)


model = pickle.load(open('./question_pred/random_forest_model.pkl', 'rb'))
vectorizer = pickle.load(open('./question_pred/vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('./question_pred/label_encoder2.pkl', 'rb'))

@app.route('/', methods=['POST'])
def predict_difficulty():
    data = request.json
    question = data.get('question')
    options = data.get('options')
    
    if not question or not options or len(options) != 4:
        return jsonify({"error": "Invalid input. Provide a question and exactly four options."}), 400
    
    options_combined = question + ' ' + ' '.join(options)
    
    X_new = vectorizer.transform([options_combined])
    
    pred = model.predict(X_new)
    
    difficulty = label_encoder.inverse_transform(pred)
    
    return jsonify({"difficulty": difficulty[0]})

if __name__ == "__main__":
    app.run(debug=True)
