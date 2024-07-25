import numpy as np
from collections import defaultdict, Counter
from autocorrect import Speller
from pattern.text.en import singularize
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from nltk.corpus import wordnet

spell = Speller(lang='en')

# Load pre-trained word2vec model
word2vec_model = api.load('word2vec-google-news-300')

def expand_query(query):
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    expanded_query = query + " " + " ".join(synonyms)
    return expanded_query

def semantic_search(query, dictionary):
    words = query.lower().split()
    similar_words = {}
    for word in words:
        try:
            similar_words[word] = word2vec_model.most_similar(word, topn=5)
        except KeyError:
            print(f"Key '{word}' not present in vocabulary")
            similar_words[word] = []
    return similar_words


def stem_data(data, stopwords, dictionary):
    stem_table = {}
    for word in data:
        if word not in stopwords:
            stem = singularize(spell(word))
            if stem not in dictionary:
                dictionary[stem] = 0
            dictionary[stem] += 1
            stem_table[word] = stem
    return stem_table

def update_core_tables2(data, dictionary, url_map, arr_url, hash_category,
                        hash_related, hash_see, stem_table, category, url,
                        url_ID, stopwords, related, see, word_pairs,
                        word_hash, word2_hash, word2_pairs):
    for word in data:
        if word in stopwords:
            continue
        stem = stem_table[word]
        if stem not in word_hash:
            word_hash[stem] = []
        word_hash[stem].append(url_ID)
        if stem not in url_map:
            url_map[stem] = []
        url_map[stem].append(url_ID)

    arr_url.append(url)
    for cat in category:
        if cat not in hash_category:
            hash_category[cat] = []
        hash_category[cat].append(url_ID)
    
    for rel in related:
        if rel not in hash_related:
            hash_related[rel] = []
        hash_related[rel].append(url_ID)
    
    for s in see:
        if s not in hash_see:
            hash_see[s] = []
        hash_see[s].append(url_ID)

    # Update word pairs and word2_hash
    for i, word1 in enumerate(data):
        if word1 in stopwords:
            continue
        for j in range(i + 1, len(data)):
            word2 = data[j]
            if word2 in stopwords:
                continue
            pair = (stem_table[word1], stem_table[word2])
            if pair not in word_pairs:
                word_pairs[pair] = 0
            word_pairs[pair] += 1

            if stem_table[word1] not in word2_hash:
                word2_hash[stem_table[word1]] = {}
            if stem_table[word2] not in word2_hash[stem_table[word1]]:
                word2_hash[stem_table[word1]][stem_table[word2]] = 0
            word2_hash[stem_table[word1]][stem_table[word2]] += 1

            if pair not in word2_pairs:
                word2_pairs[pair] = 0
            word2_pairs[pair] += 1

    return url_ID + 1

def create_pmi_table(word_pairs, dictionary):
    total_pairs = sum(word_pairs.values())
    pmi_table = {}
    for pair in word_pairs:
        word1, word2 = pair
        prob_word1 = dictionary[word1] / total_pairs
        prob_word2 = dictionary[word2] / total_pairs
        prob_pair = word_pairs[pair] / total_pairs
        pmi_table[pair] = np.log(prob_pair / (prob_word1 * prob_word2))
    return pmi_table

def create_embeddings(word_hash, pmi_table):
    embeddings = {}
    for word in word_hash:
        embedding = defaultdict(float)
        for pair in pmi_table:
            if word in pair:
                other_word = pair[0] if pair[1] == word else pair[1]
                embedding[other_word] += pmi_table[pair]
        embeddings[word] = embedding
    return embeddings

def build_ngrams(dictionary, n=4):
    ngrams_table = defaultdict(int)
    for word in dictionary:
        for i in range(len(word) - n + 1):
            ngram = word[i:i+n]
            ngrams_table[ngram] += 1
    return ngrams_table

def compress_ngrams(dictionary, ngrams_table, min_count=5):
    compressed_ngrams_table = {ngram: count for ngram, count in ngrams_table.items() if count >= min_count}
    return compressed_ngrams_table

def compress_word2_hash(dictionary, word2_hash, min_count=5):
    compressed_word2_hash = {}
    for word1 in word2_hash:
        compressed_word2_hash[word1] = {word2: count for word2, count in word2_hash[word1].items() if count >= min_count}
    return compressed_word2_hash