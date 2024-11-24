# Adaptive Quiz API

This repository hosts an adaptive quiz API built with Flask and machine learning. The API enables quizzes to be responsive to user performance, adjusting question difficulty and providing personalized feedback.

## Project Overview

The Adaptive Quiz API offers three main functionalities:

1. **Difficulty Prediction**: Predicts the difficulty level of a question based on the question text and options.
2. **Next Question Prediction**: Determines the appropriate difficulty of the next question based on user performance.
3. **Performance Feedback Generation**: Provides personalized feedback based on user metrics, clustering, and comparisons with average metrics.

The machine learning models are developed with `scikit-learn` and deployed on [Render](https://render.com).

## API Endpoints

The API is hosted at [https://quizz-ml.onrender.com](https://quizz-ml.onrender.com).

### 1. Predict Next Question Difficulty

- **Endpoint**: `/next`
- **Method**: `POST`
- **Description**: Predicts the difficulty level of the next question based on the user's recent quiz performance.
  
  - **Expected JSON Input**:
    ```json
    [{
      "user_streak": int,
      "last_difficulty": "level(easy,medium,hard)",
      "time_taken": int,
      "is_correct": int(1 or 0)
    }]
    ```
  - **Example Request**:
    ```json
    [{
      "user_streak": 3,
      "last_difficulty": "medium",
      "time_taken": 45,
      "is_correct": 1
    }]
    ```
  - **Example Response**:
    ```json
    {
      "Prediction": ["hard"]
    }
    ```

### 2. Predict Question Difficulty

- **Endpoint**: `/pred`
- **Method**: `POST`
- **Description**: Predicts the difficulty level of a multiple-choice question based on the question text and options provided.

  - **Expected JSON Input**:
    ```json
    {
      "question_text": "string",
      "option_1": "string",
      "option_2": "string",
      "option_3": "string",
      "option_4": "string"
    }
    ```
  - **Example Request**:
    ```json
    {
      "question_text": "Who was the first human to travel into space?",
      "option_1": "Neil Armstrong",
      "option_2": "Yuri Gagarin",
      "option_3": "Buzz Aldrin",
      "option_4": "Alan Shepard"
    }
    ```
  - **Example Response**:
    ```json
    {
      "difficulty": "easy"
    }
    ```

### 3. Generate Performance Feedback

- **Endpoint**: `/sum`
- **Method**: `POST`
- **Description**: Generates personalized feedback for users based on quiz performance metrics, clustering, and comparison with averages.
  
  - **Expected JSON Input**:
    ```json
    {
      "user_data": [
        {
          "username": "string",
          "score": int,
          "correct_answers": int,
          "incorrect_answers": int,
          "streak": int
        }
      ],
      "averages": {
        "average_score": float,
        "average_correct_answers": float,
        "average_incorrect_answers": float,
        "average_streak": float
      }
    }
    ```
  - **Example Request**:
    ```json
    {
      "user_data": [
        {
          "username": "john_doe",
          "score": 85,
          "correct_answers": 8,
          "incorrect_answers": 2,
          "streak": 5
        }
      ],
      "averages": {
        "average_score": 75.5,
        "average_correct_answers": 7.2,
        "average_incorrect_answers": 3.1,
        "average_streak": 4.0
      }
    }
    ```
  - **Example Response**:
    ```json
    {
      "feedback": [
        "Great work, John! Your performance is strong, with accuracy above the average. Keep pushing for perfection!"
      ],
      "analysis": [
        {
          "username": "john_doe",
          "analysis": "Your score is above average, with a strong streak. Work on reducing errors for even higher accuracy."
        }
      ],
      "graph": "more than 70,000+words"
    }
    ```

    To display the graph in HTML, use the following code:

    ```html
    <img src="data:image/png;base64,{{ graph }}">
    ```

## Project Structure

- `app1`: Contains Flask blueprint and model for predicting the next question difficulty based on user performance.
- `app2`: Contains Flask blueprint and model for predicting question difficulty based on question text and options.
- `app3`: Contains Flask blueprint and Google Gemini integration for feedback generation.
- `model.pkl`: Model for next question difficulty prediction.
- `random_forest_model.pkl`: Model for question difficulty prediction.
- `kmeans_model.pkl`: Model for user clustering in feedback generation.
- `su_scaler.pkl`: Scaler for normalizing user performance metrics.
- `label_encoder.pkl` & `label_encoder2.pkl`: Encoders for transforming categorical labels in both models.
- `vectorizer.pkl`: TF-IDF vectorizer for processing question and options text.

## Getting Started

### Prerequisites

- Python 3.x
- `pip` (Python package installer)

### Installing Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
