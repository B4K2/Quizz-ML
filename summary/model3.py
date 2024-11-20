from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

# Load data
data = pd.read_csv('updated_fabricated_quiz_data.csv')
df = pd.DataFrame(data)

# Calculate accuracy
df['accuracy'] = df['correct_answers'] / (df['correct_answers'] + df['incorrect_answers'])

# Features to use for clustering
X = df[['score', 'correct_answers', 'incorrect_answers', 'streak', 'accuracy']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Test prediction with a new sample (scaled before prediction)
sample = scaler.transform([[2, 1, 1, 1, 0.5]])  # Include 5 features
print(kmeans.predict(sample))

# Save the model and scaler
pickle.dump(kmeans, open('kmeans_model.pkl', 'wb'))
pickle.dump(scaler, open('su_scaler.pkl', 'wb'))
