import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, Blueprint
import pickle
import os
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import logging
from flask_cors import CORS
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load environment variables
dotenv = find_dotenv()
load_dotenv(dotenv)

# Initialize the Flask Blueprint
app = Blueprint('app4', __name__)

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set to DEBUG for more verbosity
logger = logging.getLogger(__name__)

# Set base directory for models
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'kmeans_model.pkl')
scaler_path = os.path.join(base_dir, 'su_scaler.pkl')

# Load the KMeans model and scaler
try:
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")

# Configure Google Gemini API
genai.configure(api_key=os.getenv("API_KEY"))  # Replace with your actual API key

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
        logger.error(f"Error generating feedback: {str(e)}")
        return f"Error generating feedback: {str(e)}"
    
def create_graphs(user_data, averages):
    """
    Create multiple graphs with grouped bars (one for user, one for average) 
    and return the base64-encoded PNG image.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 grid of graphs
    user = user_data[0]  # Assuming single-user data; adapt if necessary
    
    # Color-changing logic
    def get_color(user_value, avg_value, reverse=False):
        if reverse:
            return 'red' if user_value > avg_value else 'green'
        return 'green' if user_value >= avg_value else 'red'

    # 1. Score vs Average Score
    axs[0, 0].bar(['User Score', 'Average Score'], 
                  [user['score'], averages['average_score']],
                  color=[get_color(user['score'], averages['average_score']), 'blue'])
    axs[0, 0].set_title('Score Comparison')
    axs[0, 0].set_ylabel('Score')

    # 2. Correct Answers vs Average Correct Answers
    axs[0, 1].bar(['User Correct', 'Average Correct'], 
                  [user['correct_answers'], averages['average_correct_answers']],
                  color=[get_color(user['correct_answers'], averages['average_correct_answers']), 'blue'])
    axs[0, 1].set_title('Correct Answers Comparison')
    axs[0, 1].set_ylabel('Correct Answers')

    # 3. Incorrect Answers vs Average Incorrect Answers (reversed logic)
    axs[1, 0].bar(['User Incorrect', 'Average Incorrect'], 
                  [user['incorrect_answers'], averages['average_incorrect_answers']],
                  color=[get_color(user['incorrect_answers'], averages['average_incorrect_answers'], reverse=True), 'blue'])
    axs[1, 0].set_title('Incorrect Answers Comparison')
    axs[1, 0].set_ylabel('Incorrect Answers')

    # 4. Streak vs Average Streak
    axs[1, 1].bar(['User Streak', 'Average Streak'], 
                  [user['streak'], averages['average_streak']],
                  color=[get_color(user['streak'], averages['average_streak']), 'blue'])
    axs[1, 1].set_title('Streak Comparison')
    axs[1, 1].set_ylabel('Streak')

    # Adjust layout
    plt.tight_layout()

    # Save to a BytesIO object
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)  # Rewind the stream to the beginning
    
    # Convert to base64
    img_base64 = base64.b64encode(img_stream.read()).decode('utf-8')
    return img_base64




@app.route('/', methods=["POST"])
def home():
    logger.info("API accessed.")
    try:
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

        graph_base64 = create_graphs(user_data, averages)
        
        return jsonify({"feedback": feedback, "analysis": analysis, "graph": graph_base64})
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500
