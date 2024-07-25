# xllm-enterprise.py

# Import necessary libraries
import os
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd
import csv
import json
from fpdf import FPDF
import multiprocessing
from textblob import TextBlob
import logging
import schedule
import time

#--- [1] Backend: functions

def update_hash(hash, key, count=1):
    if key in hash:
        hash[key] += count
    else:
        hash[key] = count
    return hash

def update_nestedHash(hash, key, value, count=1):
    if key in hash:
        local_hash = hash[key]
    else:
        local_hash = {}
    if type(value) is not tuple:
        value = (value,)
    for item in value:
        if item in local_hash:
            local_hash[item] += count
        else:
            local_hash[item] = count
        hash[key] = local_hash
    return hash

def get_value(key, hash):
    return hash.get(key, '')

def update_tables(backendTables, word, hash_crawl, backendParams):
    category = get_value('category', hash_crawl)
    tag_list = get_value('tag_list', hash_crawl)
    title = get_value('title', hash_crawl)
    description = get_value('description', hash_crawl)
    meta = get_value('meta', hash_crawl)
    ID = get_value('ID', hash_crawl)
    full_content = get_value('full_content', hash_crawl)

    extraWeights = backendParams['extraWeights']
    word = word.lower()  # add stemming
    weight = 1.0
    if word in category:
        weight += extraWeights['category']
    if word in tag_list:
        weight += extraWeights['tag_list']
    if word in title:
        weight += extraWeights['title']
    if word in meta:
        weight += extraWeights['meta']

    update_hash(backendTables['dictionary'], word, weight)
    update_nestedHash(backendTables['hash_context1'], word, category)
    update_nestedHash(backendTables['hash_context2'], word, tag_list)
    update_nestedHash(backendTables['hash_context3'], word, title)
    update_nestedHash(backendTables['hash_context4'], word, description)
    update_nestedHash(backendTables['hash_context5'], word, meta)
    update_nestedHash(backendTables['hash_ID'], word, ID)
    update_nestedHash(backendTables['full_content'], word, full_content)

    return backendTables

def clean_list(value):
    value = value.replace("[", "").replace("]", "")
    aux = value.split("~")
    value_list = ()
    for val in aux:
        val = val.replace("'", "").replace('"', "").lstrip()
        if val != '':
            value_list = (*value_list, val)
    return value_list

def get_key_value_pairs(entity):
    entity = entity[1].replace("}", ", '")
    flag = False
    entity2 = ""
    for idx in range(len(entity)):
        if entity[idx] == '[':
            flag = True
        elif entity[idx] == ']':
            flag = False
        if flag and entity[idx] == ",":
            entity2 += "~"
        else:
            entity2 += entity[idx]
    entity = entity2
    key_value_pairs = entity.split(", '")
    return key_value_pairs

def update_dict(backendTables, hash_crawl, backendParams):
    max_multitoken = backendParams['max_multitoken']
    maxDist = backendParams['maxDist']
    maxTerms = backendParams['maxTerms']

    category = get_value('category', hash_crawl)
    tag_list = get_value('tag_list', hash_crawl)
    title = get_value('title', hash_crawl)
    description = get_value('description', hash_crawl)
    meta = get_value('meta', hash_crawl)

    text = category + "." + str(tag_list) + "." + title + "." + description + "." + meta
    text = text.replace('/', " ").replace('(', ' ').replace(')', ' ').replace('?', '')
    text = text.replace("'", "").replace('"', "").replace('\\n', '').replace('!', '')
    text = text.replace("\\s", '').replace("\\t", '').replace(",", " ")
    text = text.lower()
    sentence_separators = ('.',)
    for sep in sentence_separators:
        text = text.replace(sep, '_~')
    text = text.split('_~')

    hash_pairs = backendTables['hash_pairs']
    ctokens = backendTables['ctokens']
    hwords = {}  # local word hash with word position, to update hash_pairs

    for sentence in text:
        words = sentence.split(" ")
        position = 0
        buffer = []

        for word in words:
            if word not in stopwords:
                # word is single token
                buffer.append(word)
                key = (word, position)
                update_hash(hwords, key)  # for word correlation table (hash_pairs)
                update_tables(backendTables, word, hash_crawl, backendParams)

                for k in range(1, max_multitoken):
                    if position > k:
                        # word is now multi-token with k+1 tokens
                        word = buffer[position - k] + "~" + word
                        key = (word, position)
                        update_hash(hwords, key)  # for word correlation table (hash_pairs)
                        update_tables(backendTables, word, hash_crawl, backendParams)

                position += 1

    for keyA in hwords:
        for keyB in hwords:
            wordA = keyA[0]
            positionA = keyA[1]
            n_termsA = len(wordA.split("~"))

            wordB = keyB[0]
            positionB = keyB[1]
            n_termsB = len(wordB.split("~"))

            key = (wordA, wordB)
            n_termsAB = max(n_termsA, n_termsB)
            distanceAB = abs(positionA - positionB)

            if wordA < wordB and distanceAB <= maxDist and n_termsAB <= maxTerms:
                hash_pairs = update_hash(hash_pairs, key)
                if distanceAB > 1:
                    ctokens = update_hash(ctokens, key)

    return backendTables
# Define stopwords (this list can be extended)
stopwords = {'the', 'is', 'in', 'and', 'to', 'a', 'of', 'for', 'with', 'on', 'as', 'by', 'an', 'it', 'or', 'be'}

# Function to extract and preprocess text
def preprocess_text(text, stopwords):
    text = text.replace('/', " ").replace('(', ' ').replace(')', ' ').replace('?', '')
    text = text.replace("'", "").replace('"', "").replace('\\n', '').replace('!', '')
    text = text.replace("\\s", '').replace("\\t", '').replace(",", " ")
    text = text.lower()
    sentence_separators = ('.',)
    for sep in sentence_separators:
        text = text.replace(sep, '_~')
    text = text.split('_~')

    processed_text = []
    for sentence in text:
        words = sentence.split(" ")
        processed_sentence = [word for word in words if word not in stopwords]
        processed_text.append(processed_sentence)

    return processed_text

# Function to update backend tables with processed text
def update_backend_tables(backendTables, processed_text, hash_crawl, backendParams):
    max_multitoken = backendParams['max_multitoken']
    maxDist = backendParams['maxDist']
    maxTerms = backendParams['maxTerms']

    for sentence in processed_text:
        hwords = {}
        position = 0
        buffer = []

        for word in sentence:
            buffer.append(word)
            key = (word, position)
            update_hash(hwords, key)
            update_tables(backendTables, word, hash_crawl, backendParams)

            for k in range(1, max_multitoken):
                if position > k:
                    word = buffer[position - k] + "~" + word
                    key = (word, position)
                    update_hash(hwords, key)
                    update_tables(backendTables, word, hash_crawl, backendParams)

            position += 1

        for keyA in hwords:
            for keyB in hwords:
                wordA = keyA[0]
                positionA = keyA[1]
                n_termsA = len(wordA.split("~"))

                wordB = keyB[0]
                positionB = keyB[1]
                n_termsB = len(wordB.split("~"))

                key = (wordA, wordB)
                n_termsAB = max(n_termsA, n_termsB)
                distanceAB = abs(positionA - positionB)

                if wordA < wordB and distanceAB <= maxDist and n_termsAB <= maxTerms:
                    backendTables['hash_pairs'] = update_hash(backendTables['hash_pairs'], key)
                    if distanceAB > 1:
                        backendTables['ctokens'] = update_hash(backendTables['ctokens'], key)

    return backendTables

# Initialize backend tables
def initialize_backend_tables():
    return {
        'dictionary': {},
        'hash_context1': {},
        'hash_context2': {},
        'hash_context3': {},
        'hash_context4': {},
        'hash_context5': {},
        'hash_ID': {},
        'full_content': {},
        'hash_pairs': {},
        'ctokens': {}
    }
# Function to perform clustering using KMeans
def perform_clustering(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

# Function to perform PCA for dimensionality reduction
def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data

# Function to enhance query understanding using contextual models
def contextual_query_expansion(query, model, tokenizer):
    inputs = tokenizer(query, return_tensors='pt')
    outputs = model(**inputs)
    expanded_query = tokenizer.decode(outputs.logits.argmax(dim=-1)[0])
    return expanded_query

# Function to classify content based on query
def classify_content(query, model, tokenizer, labels):
    inputs = tokenizer(query, return_tensors='pt')
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    classified_label = labels[predictions.item()]
    return classified_label

# Function to process text for embeddings
def get_text_embeddings(text, embedding_model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt')
    embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Function to retrieve related multitokens
def retrieve_related_multitokens(multitokens, embeddings, relevancy_threshold=0.5):
    related_multitokens = []
    for multitoken, embedding in zip(multitokens, embeddings):
        if embedding >= relevancy_threshold:
            related_multitokens.append(multitoken)
    return related_multitokens
# Function to scrape data from a URL
def scrape_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to retrieve data from {url}, status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error occurred while scraping data from {url}: {e}")
        return None

# Function to update dataset from an API
def update_dataset_from_api(api_url, headers=None, params=None):
    try:
        response = requests.get(api_url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve data from API {api_url}, status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error occurred while retrieving data from API {api_url}: {e}")
        return None

# Function to preprocess and integrate new data
def preprocess_and_integrate_data(new_data, backendTables, backendParams):
    for entry in new_data:
        processed_text = preprocess_text(entry, stopwords)
        backendTables = update_backend_tables(backendTables, processed_text, entry, backendParams)
    return backendTables

# Function to automate data processing
def automate_data_processing(api_url, backendTables, backendParams, headers=None, params=None):
    new_data = update_dataset_from_api(api_url, headers, params)
    if new_data:
        backendTables = preprocess_and_integrate_data(new_data, backendTables, backendParams)
    return backendTables

# Function to export results in CSV format
def export_to_csv(data, file_name='results.csv'):
    keys = data[0].keys()
    with open(file_name, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)
    print(f"Data successfully exported to {file_name}")

# Function to export results in JSON format
def export_to_json(data, file_name='results.json'):
    with open(file_name, 'w') as output_file:
        json.dump(data, output_file, indent=4)
    print(f"Data successfully exported to {file_name}")

# Function to export results in PDF format
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Search Results', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(10)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

def export_to_pdf(data, file_name='results.pdf'):
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title('Search Results')
    for entry in data:
        for key, value in entry.items():
            pdf.chapter_title(key)
            pdf.chapter_body(str(value))
    pdf.output(file_name)
    print(f"Data successfully exported to {file_name}")

# Function to generate a detailed report
def generate_report(data, file_format='csv'):
    if file_format == 'csv':
        export_to_csv(data)
    elif file_format == 'json':
        export_to_json(data)
    elif file_format == 'pdf':
        export_to_pdf(data)
    else:
        print("Unsupported file format. Please choose 'csv', 'json', or 'pdf'.")

# Example data to export
example_data = [
    {"title": "Document 1", "content": "This is the content of document 1."},
    {"title": "Document 2", "content": "This is the content of document 2."}
]

# Generate reports in different formats
generate_report(example_data, file_format='csv')
generate_report(example_data, file_format='json')
generate_report(example_data, file_format='pdf')

# Function to generate and display a word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Function to plot embeddings using PCA
def plot_embeddings(embeddings, labels=None):
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(embeddings)
    df = pd.DataFrame(transformed_data, columns=['PCA1', 'PCA2'])
    if labels is not None:
        df['Label'] = labels
        fig = px.scatter(df, x='PCA1', y='PCA2', color='Label', title='PCA Embeddings')
    else:
        fig = px.scatter(df, x='PCA1', y='PCA2', title='PCA Embeddings')
    fig.show()

# Function to create a distribution plot
def plot_distribution(data, title='Distribution Plot'):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=30, alpha=0.75, edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Example usage of visualization functions
example_text = "Data science is an inter-disciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data. Data science is related to data mining, machine learning and big data."

# Generate a word cloud
generate_word_cloud(example_text)

# Example embeddings (randomly generated for demonstration purposes)
example_embeddings = np.random.rand(100, 50)  # 100 samples with 50 features each

# Plot embeddings using PCA
plot_embeddings(example_embeddings)

# Example data for distribution plot (randomly generated for demonstration purposes)
example_data = np.random.randn(1000)

# Create a distribution plot
plot_distribution(example_data)

# Function for parallel processing
def parallel_processing(data, function, n_jobs=4):
    pool = multiprocessing.Pool(n_jobs)
    results = pool.map(function, data)
    pool.close()
    pool.join()
    return results

# Example function to process data in parallel
def example_function(data_point):
    # Simulate some processing
    return data_point ** 2

# Function to optimize data structures
def optimize_data_structures(data):
    return np.array(data)

# Function for sentiment analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment

# Function for topic modeling using Latent Dirichlet Allocation (LDA)
def perform_topic_modeling(texts, n_topics=5, n_words=10):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(dtm)

    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topics.append([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-n_words - 1:-1]])
    return topics

# Example data for parallel processing
example_data = [1, 2, 3, 4, 5]

# Perform parallel processing
parallel_results = parallel_processing(example_data, example_function)
print("Parallel Processing Results:", parallel_results)

# Optimize data structures
optimized_data = optimize_data_structures(example_data)
print("Optimized Data Structure:", optimized_data)

# Example text for sentiment analysis
example_text = "I love this product! It has really made my life better."

# Analyze sentiment
sentiment = analyze_sentiment(example_text)
print("Sentiment Analysis:", sentiment)

# Example texts for topic modeling
example_texts = [
    "Data science involves using machine learning techniques.",
    "Machine learning is a subset of artificial intelligence.",
    "Artificial intelligence is transforming industries.",
    "Data analysis is crucial for decision making.",
    "Decision making can be improved with data insights."
]

# Perform topic modeling
topics = perform_topic_modeling(example_texts)
print("Topic Modeling Results:", topics)

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='xllm_enterprise.log', filemode='w')

# Function to handle errors
def handle_error(e, context=""):
    logging.error(f"Error in {context}: {str(e)}")
    print(f"An error occurred in {context}. Check the log for more details.")

# Example function with error handling
def example_error_handling_function(data):
    try:
        result = data / 0  # Intentional error to demonstrate handling
    except Exception as e:
        handle_error(e, context="example_error_handling_function")
        result = None
    return result

# Function to log process start
def log_process_start(process_name):
    logging.info(f"Starting process: {process_name}")
    print(f"Starting process: {process_name}")

# Function to log process end
def log_process_end(process_name):
    logging.info(f"Ending process: {process_name}")
    print(f"Ending process: {process_name}")

# Example usage of logging
log_process_start("Example Process")

# Example data for error handling function
example_data = 10

# Process data with error handling
result = example_error_handling_function(example_data)
print("Processing result:", result)

log_process_end("Example Process")

# Function to integrate with a third-party API (e.g., Google NLP)
def integrate_with_google_nlp(text, api_key):
    url = f"https://language.googleapis.com/v1/documents:analyzeSentiment?key={api_key}"
    document = {
        "document": {
            "type": "PLAIN_TEXT",
            "content": text
        },
        "encodingType": "UTF8"
    }
    try:
        response = requests.post(url, json=document)
        if response.status_code == 200:
            return response.json()
        else:
            handle_error(f"Failed to integrate with Google NLP API, status code: {response.status_code}")
            return None
    except Exception as e:
        handle_error(e, context="integrate_with_google_nlp")
        return None

# Function to enrich data with additional information
def enrich_data_with_external_sources(data, api_key):
    enriched_data = []
    for entry in data:
        enriched_entry = entry.copy()
        sentiment_analysis = integrate_with_google_nlp(entry['content'], api_key)
        if sentiment_analysis:
            enriched_entry['sentiment'] = sentiment_analysis['documentSentiment']
        enriched_data.append(enriched_entry)
    return enriched_data

# Function to schedule periodic data updates
def schedule_periodic_updates(interval, api_url, backendTables, backendParams, headers=None, params=None):
    schedule.every(interval).minutes.do(lambda: automate_data_processing(api_url, backendTables, backendParams, headers, params))
    while True:
        schedule.run_pending()
        time.sleep(1)

# Example data to enrich
example_data = [
    {"title": "Document 1", "content": "This is the content of document 1."},
    {"title": "Document 2", "content": "This is the content of document 2."}
]

# Google NLP API Key (placeholder, replace with actual key)
api_key = "YOUR_GOOGLE_NLP_API_KEY"

# Enrich data with external sources
enriched_data = enrich_data_with_external_sources(example_data, api_key)
print("Enriched Data:", enriched_data)

# Schedule periodic updates (example: every 30 minutes)
# schedule_periodic_updates(30, "https://api.example.com/data", backendTables, backendParams)

if __name__ == "__main__":
    # Initialize backend tables and parameters
    backendTables = initialize_backend_tables()
    backendParams = {
        'extraWeights': {'category': 1.0, 'tag_list': 0.8, 'title': 1.2, 'meta': 0.5},
        'max_multitoken': 3,
        'maxDist': 5,
        'maxTerms': 3
    }

    # Example data to process
    example_data = [
        {"title": "Document 1", "content": "This is the content of document 1."},
        {"title": "Document 2", "content": "This is the content of document 2."}
    ]

    # Update backend tables with example data
    for entry in example_data:
        processed_text = preprocess_text(entry['content'], stopwords)
        backendTables = update_backend_tables(backendTables, processed_text, entry, backendParams)

    # Perform clustering on example embeddings (randomly generated for demonstration purposes)
    example_embeddings = np.random.rand(100, 50)
    labels, cluster_centers = perform_clustering(example_embeddings)

    # Perform PCA on example embeddings
    pca_transformed = perform_pca(example_embeddings)

    # Generate and display a word cloud for example text
    generate_word_cloud(" ".join([entry['content'] for entry in example_data]))

    # Example text for sentiment analysis
    example_text = "I love this product! It has really made my life better."
    sentiment = analyze_sentiment(example_text)
    print("Sentiment Analysis:", sentiment)

    # Perform topic modeling on example texts
    example_texts = [
        "Data science involves using machine learning techniques.",
        "Machine learning is a subset of artificial intelligence.",
        "Artificial intelligence is transforming industries.",
        "Data analysis is crucial for decision making.",
        "Decision making can be improved with data insights."
    ]
    topics = perform_topic_modeling(example_texts)
    print("Topic Modeling Results:", topics)

    # Enrich data with Google NLP API
    api_key = "YOUR_GOOGLE_NLP_API_KEY"
    enriched_data = enrich_data_with_external_sources(example_data, api_key)
    print("Enriched Data:", enriched_data)

    # Generate reports in different formats
    generate_report(example_data, file_format='csv')
    generate_report(example_data, file_format='json')
    generate_report(example_data, file_format='pdf')

    # Example usage of logging and error handling
    log_process_start("Example Process")
    result = example_error_handling_function(10)
    print("Processing result:", result)
    log_process_end("Example Process")

    # Schedule periodic updates (example: every 30 minutes)
    # schedule_periodic_updates(30, "https://api.example.com/data", backendTables, backendParams)
