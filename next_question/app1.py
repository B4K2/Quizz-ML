import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Blueprint
import pickle
import os


app = Blueprint('app1', __name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.pkl')
label_encoder_path = os.path.join(base_dir, 'label_encoder.pkl')


model = pickle.load(open(model_path, "rb"))
label_encoder = pickle.load(open(label_encoder_path, "rb"))

@app.route('/', methods=["POST"])
def home():
    json_ = request.json
    query_df = pd.DataFrame(json_)

    if "previous_questions" in query_df.columns:
        try:
            # Attempt to drop 'previous_questions' column
            query_df = query_df.drop(columns=["previous_questions"])
        except KeyError as e:
            # If the column is not found, catch the exception and handle it
            return jsonify({"error": f"Error in dropping 'previous_questions': {str(e)}"}), 400


    if "last_difficulty" in query_df.columns:
        try:
            query_df["last_difficulty"] = label_encoder.transform(query_df["last_difficulty"])
        except ValueError as e:
            return jsonify({"error": f"Error in transforming 'last_difficulty': {str(e)}"}), 400

    pred = model.predict(query_df)
    pred = [int(x) for x in pred]
    decoded_pred = label_encoder.inverse_transform(pred)

    return jsonify({"Prediction": list(decoded_pred)})

