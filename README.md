# Adaptive Quiz API

This repository hosts an adaptive quiz API built with Flask and machine learning. The API provides two main functionalities: predicting the difficulty level of multiple-choice questions and suggesting the difficulty of the next question based on user performance.

## Project Overview

This project is designed to make quizzes adaptive to user performance. It includes:

1. **Difficulty Prediction**: Predicts the difficulty level of a question based on the text and options provided.
2. **Next Question Prediction**: Determines the next question's difficulty based on user performance metrics, such as streak, time taken, and correctness.

The machine learning models are trained using `scikit-learn`, and the API is deployed on Render.

## API Endpoints

The API is hosted at [https://quizz-ml.onrender.com](https://quizz-ml.onrender.com).

### 1. Predict Next Question Difficulty

- **Endpoint**: [https://quizz-ml.onrender.com/next](https://quizz-ml.onrender.com/next)
- **Method**: `POST`
- **Description**: This endpoint predicts the difficulty level of the next question based on the user's recent quiz performance.
  
  - **Expected JSON Input**:
    ```json
    [{
      "user_streak": int,
      "last_difficulty": "level(easy,medium,hard)",
      "time_taken": int,
      "is_correct": int(1,0)
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

- **Endpoint**: [https://quizz-ml.onrender.com/pred](https://quizz-ml.onrender.com/pred)
- **Method**: `POST`
- **Description**: This endpoint predicts the difficulty of a given multiple-choice question based on the question text and options.

  - **Expected JSON Input**:
    ```json
    {
    "question_text": "question(string)",
    "option_1": "option",
    "option_2": "option",
    "option_3": "option",
    "option_4": "option"
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

## Project Structure

- `app1`: Contains Flask blueprint and model for predicting the next question difficulty based on user performance.
- `app2`: Contains Flask blueprint and model for predicting question difficulty based on question text and options.
- `model.pkl`: Model for next question difficulty prediction.
- `random_forest_model.pkl`: Model for question difficulty prediction.
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
