import openai
import aiohttp
import asyncio
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import logging
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize models for embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_model = AutoModel.from_pretrained("bert-base-uncased")

# Example API keys (replace with actual keys)
openai_api_key = 'sk-proj-aQ936mI4Cr2rCav7dAYvT3BlbkFJWY7zAkX5DO2Ygqd8gH8f'
mistral_api_key = 'XneQSSWXW07FeD2bslrB9NWwlPaffRzF'
perplexity_api_key = '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'

# Set up OpenAI
openai.api_key = openai_api_key

try:
    import language_tool_python
    tool = language_tool_python.LanguageTool('en-US')
    grammar_check_available = True
except ModuleNotFoundError as e:
    logging.warning(f"LanguageTool not available: {e}. Grammar checking will be limited.")
    grammar_check_available = False

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

def rate_response(prompt, response):
    # Sentiment analysis
    sentiment = TextBlob(response).sentiment.polarity

    # Grammar checking
    if grammar_check_available:
        matches = tool.check(response)
        grammar_score = max(0, 1 - len(matches) / max(1, len(response.split())))
    else:
        grammar_score = 1  # Default to 1 if grammar checking is not available

    # Semantic similarity
    similarity = np.dot(get_internal_embedding(prompt), get_internal_embedding(response).T).item()

    # Combine scores to form a rating
    rating = 0.5 * sentiment + 0.3 * grammar_score + 0.2 * similarity
    rating = max(1, min(5, int(rating * 5)))  # Scale to a 1-5 rating
    return rating

async def gather_responses(prompts):
    data = []

    async with aiohttp.ClientSession() as session:
        for prompt in prompts:
            tasks = [
                query_openai(prompt),
                query_mistral(session, prompt),
                query_perplexity(session, prompt)
            ]
            results = await asyncio.gather(*tasks)

            for response, confidence in results:
                length = len(response)
                similarity = np.dot(get_internal_embedding(prompt), get_internal_embedding(response).T)
                rating = rate_response(prompt, response)
                data.append([prompt, response, confidence, length, similarity, rating])

    return data

# Expanded list of prompts
prompts = [
    "Explain the significance of quantum computing.",
    "What are the applications of machine learning in healthcare?",
    "How does blockchain technology work?",
    "What are the benefits of renewable energy?",
    "Describe the process of photosynthesis.",
    "What is the impact of climate change on polar bears?",
    "How do vaccines work?",
    "Explain the theory of relativity.",
    "What are the effects of deforestation?",
    "How does the internet work?",
    # Add more prompts as needed
]

data = asyncio.run(gather_responses(prompts))

# Create DataFrame and save to CSV
df = pd.DataFrame(data, columns=['prompt', 'response', 'confidence', 'length', 'similarity', 'rating'])
df.to_csv('llm_responses.csv', index=False)
