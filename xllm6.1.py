import numpy as np
import xllm6_util as llm6
from autocorrect import Speller
from pattern.text.en import singularize
import os
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify, render_template
import requests
import concurrent.futures
import json

spell = Speller(lang='en')

# --- [1] some utilities

stopwords = (
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", 
    "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", 
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", 
    "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", 
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", 
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", 
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", 
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", 
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
)

utf_map = {
    '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2013': '-', '\u2014': '-', '\u2026': '...'
}

app = Flask(__name__)

def get_top_category(page):
    read = (page.split("<ul class=\"breadcrumb\">"))[1]
    read = (read.split("\">"))[1]
    top_category = (read.split("</a>"))[0]
    return top_category

def trim(word):
    return word.replace(".", "").replace(",", "")

def split_page(row):
    line = row.split("<!-- Begin Content -->")
    header = (line[0]).split("\t~")
    header = header[0]
    html = (line[1]).split("<!-- End Content -->")
    content = html[0]
    related = (html[1]).split("<h2>See also</h2>")
    if len(related) > 1:
        related = (related[1]).split("<!-- End See Also -->")
        related = related[0]
    else:
        related = ""
    see = row.split("<p class=\"CrossRefs\">")
    if len(see) > 1:
        see = (see[1]).split("<!-- Begin See Also -->")
        see = see[0]
    else:
        see = ""
    return(header, content, related, see)

def list_to_text(list):
    text = " " + str(list) + " "
    text = text.replace("'", " ")
    text = text.replace("\"", " ")
    text = text.replace("(", "( ")
    text = text.replace(")", ". )").replace(" ,", ",")
    text = text.replace(" |", ",")
    text = text.replace(" .", ".")
    text = text.lower()
    return text

print("Reading crawl data...")
file_path = "crawl_final_stats.txt"
if not os.path.exists(file_path):
    print(f"File {file_path} not found.")
    exit()

file_html = open(file_path, "r", encoding="utf-8")
Lines = file_html.readlines()
print("Crawl data read successfully.")

dictionary = {}
word_pairs = {}
url_map = {}
arr_url = []
hash_category = {}
hash_related = {}
hash_see = {}
word_hash = {}
word2_hash = {}
word2_pairs = {}

url_ID = 0

def process_row(row):
    global url_ID
    category = {}

    print("Processing row...")
    for key in utf_map:
        row = row.replace(key, utf_map[key])

    (header, content, related, see) = split_page(row)
    url = (header.split("\t"))[0]
    cat = (header.split("\t"))[1]
    cat = cat.replace(",", " |").replace("(","").replace(")","")
    cat = cat.replace("'", "").replace("\"", "")
    category[cat] = 1

    top_category = get_top_category(row)

    list_related = related.split("\">")
    related = ()
    for item in list_related:
        item = (item.split("<"))[0]
        if item != "" and "mathworld" not in item.lower():
            related = (*related, item)

    if see != "":
        list_see = see.split("\">")
        see = ()
        for item in list_see:
            item = (item.split("<"))[0]
            if item != "" and item != " ":
                see = (*see, item)

    text_category = list_to_text(category)
    text_related = list_to_text(related)
    text_see = list_to_text(see)
    content += text_category + text_related + text_see

    flag = 0
    cleaned_content = ""
    for char in content:
        if char == "\"":
            flag = 1 - flag
        if flag == 0:
            cleaned_content += char

    cleaned_content = cleaned_content.replace(">", "> ")
    cleaned_content = cleaned_content.replace("<", ". <")
    cleaned_content = cleaned_content.replace("(", "( ")
    cleaned_content = cleaned_content.lower()
    data = cleaned_content.split(" ")
    stem_table = llm6.stem_data(data, stopwords, dictionary)

    url_ID = llm6.update_core_tables2(data, dictionary, url_map, arr_url, hash_category,
                                      hash_related, hash_see, stem_table, category, url,
                                      url_ID, stopwords, related, see, word_pairs,
                                      word_hash, word2_hash, word2_pairs)
    print(f"Processed row: {url_ID}")

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_row, line) for line in Lines]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"Error occurred: {e}")

print("Creating PMI tables...")
pmi_table = llm6.create_pmi_table(word_pairs, dictionary)
pmi_table2 = llm6.create_pmi_table(word2_pairs, dictionary)
print("PMI tables created.")
print("Creating embeddings...")
embeddings = llm6.create_embeddings(word_hash, pmi_table)
ngrams_table = llm6.build_ngrams(dictionary)
compressed_ngrams_table = llm6.compress_ngrams(dictionary, ngrams_table)
compressed_word2_hash = llm6.compress_word2_hash(dictionary, word2_hash)
embeddings2 = llm6.create_embeddings(compressed_word2_hash, pmi_table2)
print("Embeddings created.")

def cluster_embeddings(embeddings, n_clusters=5):
    words = list(embeddings.keys())
    vectors = [list(embeddings[word].values()) for word in words]

    # Find the maximum length of embeddings
    max_length = max(len(vector) for vector in vectors)
    vectors = [vector + [0] * (max_length - len(vector)) for vector in vectors]

    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(vectors)
    clustered_data = [[] for _ in range(n_clusters)]

    for word, vector, cluster in zip(words, vectors, clusters):
        clustered_data[cluster].append(vector)

    return clustered_data

def visualize_embeddings(embeddings):
    words = list(embeddings.keys())
    vectors = [list(embeddings[word].values()) for word in words]

    if len(vectors) < 2:
        print("Not enough data points to perform PCA. Need at least 2.")
        return

    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 6))
    for i, word in enumerate(words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], marker='x', color='red')
        plt.text(reduced_vectors[i, 0] + 0.01, reduced_vectors[i, 1] + 0.01, word, fontsize=9)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Word Embeddings PCA Visualization')
    plt.show()

def main():
    while True:
        query = input("Enter queries (ex: Gaussian distribution, central moments): ")
        if not query:
            break

        tokens = query.lower().split()
        relevant_words = [word for word in tokens if word in dictionary]

        if not relevant_words:
            print("No relevant words found in the dictionary.")
            continue

        for word in relevant_words:
            print(f"Word: {word}")
            print(f"Frequency: {dictionary[word]}")
            if word in embeddings:
                print(f"Embedding: {embeddings[word]}")

        visualize_embeddings({word: embeddings[word] for word in relevant_words if word in embeddings})
        cluster_embeddings({word: embeddings[word] for word in relevant_words if word in embeddings})

if __name__ == "__main__":
    main()

def create_pmi_table(word_pairs, dictionary):
    pmi_table = {}
    total_word_pairs = sum(word_pairs.values())
    total_words = sum(dictionary.values())

    for pair, pair_count in word_pairs.items():
        word1, word2 = pair.split()
        word1_count = dictionary[word1]
        word2_count = dictionary[word2]
        pmi = (pair_count / total_word_pairs) / ((word1_count / total_words) * (word2_count / total_words))
        pmi_table[pair] = pmi

    return pmi_table

def create_embeddings(word_hash, pmi_table):
    embeddings = {}
    for word, neighbors in word_hash.items():
        embeddings[word] = {neighbor: pmi_table.get(f"{word} {neighbor}", 0) for neighbor in neighbors}
    return embeddings

def build_ngrams(dictionary):
    ngrams_table = {}
    for word in dictionary:
        for i in range(len(word) - 1):
            ngram = word[i:i+2]
            if ngram not in ngrams_table:
                ngrams_table[ngram] = 0
            ngrams_table[ngram] += 1
    return ngrams_table

def compress_ngrams(dictionary, ngrams_table):
    compressed_ngrams_table = {}
    for ngram, count in ngrams_table.items():
        compressed_ngrams_table[ngram] = count / sum(dictionary.values())
    return compressed_ngrams_table

def compress_word2_hash(dictionary, word2_hash):
    compressed_word2_hash = {}
    for word, neighbors in word2_hash.items():
        compressed_word2_hash[word] = {neighbor: neighbors[neighbor] / dictionary[word] for neighbor in neighbors}
    return compressed_word2_hash

if __name__ == "__main__":
    print("Reading crawl data...")
    file_path = "crawl_final_stats.txt"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        exit()

    file_html = open(file_path, "r", encoding="utf-8")
    Lines = file_html.readlines()
    print("Crawl data read successfully.")

    dictionary = {}
    word_pairs = {}
    url_map = {}
    arr_url = []
    hash_category = {}
    hash_related = {}
    hash_see = {}
    word_hash = {}
    word2_hash = {}
    word2_pairs = {}

    url_ID = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_row, line) for line in Lines]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")

    print("Creating PMI tables...")
    pmi_table = create_pmi_table(word_pairs, dictionary)
    pmi_table2 = create_pmi_table(word2_pairs, dictionary)
    print("PMI tables created.")

    print("Creating embeddings...")
    embeddings = create_embeddings(word_hash, pmi_table)
    ngrams_table = build_ngrams(dictionary)
    compressed_ngrams_table = compress_ngrams(dictionary, ngrams_table)
    compressed_word2_hash = compress_word2_hash(dictionary, word2_hash)
    embeddings2 = create_embeddings(compressed_word2_hash, pmi_table2)
    print("Embeddings created.")
