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

    # Define required columns for the model
    required_columns = ["user_streak", "last_difficulty", "time_taken", "is_correct"]
    
    # Filter DataFrame to keep only required columns
    query_df = query_df[[col for col in query_df.columns if col in required_columns]]

    # Handle encoding for 'last_difficulty' if it exists in the data
    if "last_difficulty" in query_df.columns:
        try:
            query_df["last_difficulty"] = label_encoder.transform(query_df["last_difficulty"])
        except ValueError as e:
            return jsonify({"error": f"Error in transforming 'last_difficulty': {str(e)}"}), 400

    # Make predictions
    pred = model.predict(query_df)
    decoded_pred = label_encoder.inverse_transform([int(x) for x in pred])

    return jsonify({"Prediction": list(decoded_pred)})

