import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

data = pd.read_csv("fabricated_quiz_questions.csv")

df = pd.DataFrame(data)

df['options_combined'] = df['question_text'] + ' ' + df['option_1'] + ' ' + df['option_2'] + ' ' + df['option_3'] + ' ' + df['option_4']


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['options_combined'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['difficulty'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")



pickle.dump(model, open('random_forest_model.pkl', 'wb'))

pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

pickle.dump(label_encoder, open('label_encoder2.pkl', 'wb'))