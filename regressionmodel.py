import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModel
import torch
import joblib

# Load dataset
data = pd.read_csv('llm_responses.csv')

# Check for duplicates
print(f"Number of duplicates: {data.duplicated().sum()}")
data = data.drop_duplicates()

# Check for variability in the rating
print(f"Rating distribution:\n{data['rating'].value_counts()}")

# Ensure the dataset contains all necessary columns
assert all(col in data.columns for col in ['confidence', 'length', 'similarity', 'rating'])

# Feature extraction function
def extract_features(data):
    data['length'] = data['response'].apply(len)
    
    # Example similarity feature using internal embeddings
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    def get_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
        return embeddings

    def compute_similarity(row):
        prompt_emb = get_embedding(row['prompt'])
        response_emb = get_embedding(row['response'])
        similarity = np.dot(prompt_emb, response_emb.T).item()
        return similarity
    
    data['similarity'] = data.apply(compute_similarity, axis=1)
    return data

data = extract_features(data)

# Split data
X = data[['confidence', 'length', 'similarity']]
y = data['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save model for later use
joblib.dump(model, 'response_blending_model.pkl')
