from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

data = pd.read_csv('adaptive_quiz_data.csv')
df  = pd.DataFrame(data)


label_encoder = LabelEncoder()
df['last_difficulty'] = label_encoder.fit_transform(df['last_difficulty'])
df['next_difficulty'] = label_encoder.fit_transform(df['next_difficulty'])


X = df[['user_streak', 'last_difficulty', 'time_taken', 'is_correct']]
y = df['next_difficulty']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()
model.fit(X_train, y_train)

pickle.dump(model,open("model.pkl","wb"))
pickle.dump(label_encoder,open("label_encoder.pkl","wb"))