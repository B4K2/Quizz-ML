import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Blueprint
import pickle

app = Blueprint('app1', __name__)

model = pickle.load(open("./next_question/model.pkl", "rb"))
label_encoder = pickle.load(open("./next_question/label_encoder.pkl", "rb"))

@app.route('/', methods=["POST"])
def home():
    json_ = request.json
    query_df = pd.DataFrame(json_)

    if "last_difficulty" in query_df.columns:
        try:
            query_df["last_difficulty"] = label_encoder.transform(query_df["last_difficulty"])
        except ValueError as e:
            return jsonify({"error": f"Error in transforming 'last_difficulty': {str(e)}"}), 400

    pred = model.predict(query_df)
    pred = [int(x) for x in pred]
    decoded_pred = label_encoder.inverse_transform(pred)

    return jsonify({"Prediction": list(decoded_pred)})

if __name__ == "__main__":
    app.run(debug=True)
