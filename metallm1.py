import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModel
import torch
import openai
import aiohttp
import asyncio
import logging
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Example API keys (replace with actual keys)
openai_api_key = 'sk-proj-aQ936mI4Cr2rCav7dAYvT3BlbkFJWY7zAkX5DO2Ygqd8gH8f'
mistral_api_key = 'XneQSSWXW07FeD2bslrB9NWwlPaffRzF'
perplexity_api_key = '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'

# Set up OpenAI
openai.api_key = openai_api_key

# File paths
MODEL_PATH = 'response_blending_model.pkl'
DATA_PATH = 'llm_responses.csv'

# Function to train and save the model
def train_model():
    # Load dataset
    data = pd.read_csv(DATA_PATH)

    # Check for duplicates and variability
    data = data.drop_duplicates()
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
    joblib.dump(model, MODEL_PATH)

# Function to load the model
def load_model():
    if not os.path.exists(MODEL_PATH):
        train_model()
    return joblib.load(MODEL_PATH)

# Load or train the model
model = load_model()

# Initialize tokenizer and embedding model for use in the main function
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_model = AutoModel.from_pretrained("bert-base-uncased")

async def query_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message['content'].strip()
        return text, 1  # OpenAI responses don't have a confidence score
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return "OpenAI API error occurred.", 0

async def query_mistral(session, prompt):
    url = "https://api.mistral.ai/v1/completions"
    headers = {
        "Authorization": f"Bearer {mistral_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-small-latest",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "top_p": 1,
        "max_tokens": 100,
        "stream": False,
        "safe_prompt": False,
        "random_seed": 1337
    }
    try:
        async with session.post(url, headers=headers, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            return result['choices'][0]['text'], 1  # Assuming the response structure
    except Exception as e:
        logging.error(f"Mistral API error: {e}")
        return "Mistral API error occurred.", 0

async def query_perplexity(session, prompt):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {perplexity_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3-sonar-small-32k-online",
        "messages": [
            {
                "role": "system",
                "content": "Be precise and concise."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    try:
        async with session.post(url, headers=headers, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            return result['choices'][0]['message']['content'], 1  # Assuming the response structure
    except Exception as e:
        logging.error(f"Perplexity API error: {e}")
        return "Perplexity API error occurred.", 0

def get_internal_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        with torch.no_grad():
            embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1).numpy()
        return embeddings
    except Exception as e:
        logging.error(f"Internal embedding error: {e}")
        return np.zeros((1, 768))  # Return a zero vector as a fallback

async def meta_llm(prompt):
    async with aiohttp.ClientSession() as session:
        tasks = [
            query_openai(prompt),
            query_mistral(session, prompt),
            query_perplexity(session, prompt)
        ]
        results = await asyncio.gather(*tasks)
    
    response_data = []
    for response, confidence in results:
        length = len(response)
        similarity = np.dot(get_internal_embedding(prompt), get_internal_embedding(response).T)
        response_data.append([confidence, length, similarity, response])
    
    response_df = pd.DataFrame(response_data, columns=['confidence', 'length', 'similarity', 'response'])
    predictions = model.predict(response_df[['confidence', 'length', 'similarity']])
    best_response_idx = np.argmax(predictions)
    best_response = response_df.iloc[best_response_idx]['response']
    
    return best_response

async def meta_llm_batch(prompts):
    tasks = [meta_llm(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)

# Example usage
prompts = [
    "Explain the significance of quantum computing.",
    "What are the applications of machine learning in healthcare?",
    "How does blockchain technology work?"
]

responses = asyncio.run(meta_llm_batch(prompts))
for i, response in enumerate(responses):
    print(f"Response to prompt {i+1}: {response}")
