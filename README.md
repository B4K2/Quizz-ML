# Adaptive Quiz API

This project provides an adaptive quiz system with two main functionalities: predicting the difficulty level of multiple-choice questions and determining the next question's difficulty level based on a user's performance. The backend is built with Flask and uses machine learning models trained with `scikit-learn`.

## Features

1. **Difficulty Prediction**: Predicts the difficulty level (easy, medium, hard) of a multiple-choice question based on the question text and options.
2. **Next Question Prediction**: Determines the next question's difficulty level based on the user's recent quiz performance.

## Project Structure

- `app1`: Flask blueprint for predicting the next question difficulty level based on user performance.
- `app2`: Flask blueprint for predicting the difficulty of a multiple-choice question.
- `model.pkl`: Model for predicting the next question difficulty level.
- `random_forest_model.pkl`: Model for predicting question difficulty based on question text and options.
- `label_encoder.pkl` & `label_encoder2.pkl`: Encoders to transform labels in each model.
- `vectorizer.pkl`: TF-IDF vectorizer for transforming question text and options.

## Getting Started

### Prerequisites

- Python 3.x
- `pip` (Python package installer)

### Libraries Used

- Flask==3.0.3
- numpy==2.1.3
- pandas==2.2.3
- scikit-learn==1.5.2
- gunicorn==23.0.0

Install these dependencies using:

```bash
pip install -r requirements.txt
