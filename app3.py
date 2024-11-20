import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Blueprint
import pickle
import os

app = Blueprint('app3', __name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'kmeans_model.pkl')
scaler_path = os.path.join(base_dir, 'su_scaler.pkl')

# Load the model and scaler
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

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

    # Select only the numeric features for scaling (excluding 'username')
    X = query_df[['score', 'correct_answers', 'incorrect_answers', 'streak', 'accuracy']]

    # Scale the features
    X_scaled = scaler.transform(X)  # Use transform instead of fit_transform for new data

    # Predict clusters
    query_df['cluster'] = model.predict(X_scaled)

    # Generate feedback based on the cluster assignment and averages
    feedback = []
    analysis = []

    for _, row in query_df.iterrows():
        # Accuracy-Based Feedback
        if row['accuracy'] > 0.8:
            accuracy_feedback = f"Excellent work, {row['username']}! Your accuracy is {row['accuracy']:.2f}, which is outstanding."
        elif row['accuracy'] > 0.6:
            accuracy_feedback = f"Good job, {row['username']}! Your accuracy is {row['accuracy']:.2f}. Keep focusing on improvements."
        else:
            accuracy_feedback = f"Hey {row['username']}, your accuracy is {row['accuracy']:.2f}. Consider revising areas where you struggled."

        # Cluster-Based Feedback
        if row['cluster'] == 0:
            cluster_feedback = "It looks like you’re facing some challenges. Spend more time reviewing mistakes and practicing consistently."
        elif row['cluster'] == 1:
            cluster_feedback = "You're progressing well but need to work on consistency. Keep up the effort and stay focused!"
        elif row['cluster'] == 2:
            cluster_feedback = "You’re improving steadily. Keep refining your strategies to minimize errors and boost your performance."
        elif row['cluster'] == 3:
            cluster_feedback = "Great work! You're performing well. Keep up the momentum and aim for even better results."
        elif row['cluster'] == 4:
            cluster_feedback = "Outstanding performance! You're among the top performers. Keep pushing towards perfection!"

        # Compare with averages
        avg_feedback = []
        if row['score'] > averages['average_score']:
            avg_feedback.append(f"Your score of {row['score']} is above the average score of {averages['average_score']:.2f}. Great job!")
        else:
            avg_feedback.append(f"Your score of {row['score']} is below the average score of {averages['average_score']:.2f}. Focus on improving.")

        if row['correct_answers'] > averages['average_correct_answers']:
            avg_feedback.append(f"You answered {row['correct_answers']} questions correctly, above the average of {averages['average_correct_answers']:.2f}.")
        else:
            avg_feedback.append(f"You answered {row['correct_answers']} questions correctly, below the average of {averages['average_correct_answers']:.2f}. Work on improving accuracy.")

        if row['incorrect_answers'] < averages['average_incorrect_answers']:
            avg_feedback.append(f"You made {row['incorrect_answers']} incorrect answers, which is better than the average of {averages['average_incorrect_answers']:.2f}.")
        else:
            avg_feedback.append(f"You made {row['incorrect_answers']} incorrect answers, which is higher than the average of {averages['average_incorrect_answers']:.2f}. Aim to reduce mistakes.")

        if row['streak'] > averages['average_streak']:
            avg_feedback.append(f"Your streak of {row['streak']} is above the average streak of {averages['average_streak']:.2f}. Keep it up!")
        else:
            avg_feedback.append(f"Your streak of {row['streak']} is below the average streak of {averages['average_streak']:.2f}. Try to maintain longer streaks.")

        # Combine all feedback
        feedback_msg = f"{accuracy_feedback} {cluster_feedback}"
        avg_msg = " ".join(avg_feedback)

        feedback.append(feedback_msg)
        analysis.append({"username": row['username'], "analysis": avg_msg})
    
    return jsonify({"feedback": feedback, "analysis": analysis})
