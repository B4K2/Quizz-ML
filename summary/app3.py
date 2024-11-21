import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Blueprint
import pickle
import os
import google.generativeai as genai
from dotenv import load_dotenv


# Initialize the Flask Blueprint
app = Blueprint('app3', __name__)

# Set base directory for models
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'kmeans_model.pkl')
scaler_path = os.path.join(base_dir, 'su_scaler.pkl')

# Load the KMeans model and scaler
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# Configure Google Gemini API
genai.configure(api_key="AIzaSyA5_1uvb5-fhZelP-DZkzvCW0TDFC0RByg")  # Replace with your actual API key

def generate_gemini_feedback(username, metrics, cluster_feedback, avg_feedback):
    """
    Generate feedback using Google Gemini API.
    """
    prompt = f"""
    Provide small paragraph type feedback for a quiz participant named {username}.
    Metrics:
    - Score: {metrics['score']}
    - Correct Answers: {metrics['correct_answers']}
    - Incorrect Answers: {metrics['incorrect_answers']}
    - Streak: {metrics['streak']}
    - Accuracy: {metrics['accuracy']:.2f}

    Cluster Feedback:
    {cluster_feedback}

    Comparison to Averages:
    {avg_feedback}

    Feedback should highlight strengths, suggest improvement areas, and remain encouraging.
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating feedback: {str(e)}"

@app.route('/', methods=["POST"])
def home():
    # Parse the JSON request
    request_data = request.json
    
    # Extract user data and average metrics
    user_data = request_data['user_data']
    averages = request_data['averages']

    # Convert user data to a DataFrame
    query_df = pd.DataFrame(user_data)

    # Calculate accuracy for each user
    query_df['accuracy'] = query_df['correct_answers'] / (query_df['correct_answers'] + query_df['incorrect_answers'])

    # Select numeric features for scaling
    X = query_df[['score', 'correct_answers', 'incorrect_answers', 'streak', 'accuracy']]
    X_scaled = scaler.transform(X)

    # Predict clusters
    query_df['cluster'] = model.predict(X_scaled)

    # Generate feedback based on cluster and averages
    feedback = []
    analysis = []

    for _, row in query_df.iterrows():
        # Cluster-Based Feedback
        cluster_feedback = {
            0: "It looks like you're facing some challenges. Spend more time reviewing mistakes and practicing consistently.",
            1: "You're progressing well but need to work on consistency. Keep up the effort and stay focused!",
            2: "You’re improving steadily. Keep refining your strategies to minimize errors and boost performance.",
            3: "Great work! You're performing well. Keep up the momentum and aim for even better results.",
            4: "Outstanding performance! You're among the top performers. Keep pushing towards perfection!",
        }.get(row['cluster'], "Cluster information not available.")

        # Construct average feedback messages
        avg_feedback = []
        if row['score'] > averages['average_score']:
            avg_feedback.append(f"Your score of {row['score']} is above the average score of {averages['average_score']:.2f}.")
        else:
            avg_feedback.append(f"Your score of {row['score']} is below the average score of {averages['average_score']:.2f}.")

        if row['correct_answers'] > averages['average_correct_answers']:
            avg_feedback.append(f"You answered {row['correct_answers']} questions correctly, above the average of {averages['average_correct_answers']:.2f}.")
        else:
            avg_feedback.append(f"You answered {row['correct_answers']} questions correctly, below the average of {averages['average_correct_answers']:.2f}.")

        if row['incorrect_answers'] < averages['average_incorrect_answers']:
            avg_feedback.append(f"You made {row['incorrect_answers']} incorrect answers, better than the average of {averages['average_incorrect_answers']:.2f}.")
        else:
            avg_feedback.append(f"You made {row['incorrect_answers']} incorrect answers, higher than the average of {averages['average_incorrect_answers']:.2f}.")

        if row['streak'] > averages['average_streak']:
            avg_feedback.append(f"Your streak of {row['streak']} is above the average streak of {averages['average_streak']:.2f}.")
        else:
            avg_feedback.append(f"Your streak of {row['streak']} is below the average streak of {averages['average_streak']:.2f}.")

        avg_feedback_str = " ".join(avg_feedback)

        # Metrics for Gemini prompt
        metrics = {
            "score": row['score'],
            "correct_answers": row['correct_answers'],
            "incorrect_answers": row['incorrect_answers'],
            "streak": row['streak'],
            "accuracy": row['accuracy'],
        }

        # Generate feedback with Google Gemini
        user_feedback = generate_gemini_feedback(row['username'], metrics, cluster_feedback, avg_feedback_str)

        feedback.append(user_feedback)
        analysis.append({"username": row['username'], "analysis": avg_feedback_str})
    
    return jsonify({"feedback": feedback, "analysis": analysis})
