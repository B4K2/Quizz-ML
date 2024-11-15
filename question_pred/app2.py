from flask import Flask, request, jsonify, Blueprint
import numpy
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os


app = Blueprint('app2', __name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'random_forest_model.pkl')
vectorizer_path = os.path.join(base_dir, 'vectorizer.pkl')
label_encoder_path = os.path.join(base_dir, 'label_encoder2.pkl')


model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))
label_encoder = pickle.load(open(label_encoder_path, 'rb'))

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

