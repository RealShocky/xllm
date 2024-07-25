from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
from functools import wraps
import xllm6_util as llm6
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import asyncio
from metallm1 import meta_llm_batch
import logging

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'your_secret_key'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

FEATURE_FLAGS = {
    'free': ['basic_search'],
    'pro': ['basic_search', 'advanced_search', 'data_export'],
    'enterprise': ['basic_search', 'advanced_search', 'data_export', 'ml_integration']
}

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    tier = db.Column(db.String(50), nullable=False)
    query_count = db.Column(db.Integer, default=0)
    query_reset_date = db.Column(db.DateTime, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def tier_required(tier):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if current_user.tier != tier:
                abort(403, 'Feature not available in your tier')
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def feature_enabled(feature):
    return feature in FEATURE_FLAGS.get(current_user.tier, [])

def reset_query_counts():
    users = User.query.all()
    for user in users:
        # Ensure user.query_reset_date is timezone-aware
        if user.query_reset_date.tzinfo is None:
            user.query_reset_date = user.query_reset_date.replace(tzinfo=timezone.utc)
        if user.query_reset_date < datetime.now(timezone.utc):
            user.query_count = 0
            user.query_reset_date = datetime.now(timezone.utc) + timedelta(days=30)
            db.session.commit()


@app.before_request
def before_request():
    reset_query_counts()
    if current_user.is_authenticated:
        if current_user.tier == 'free' and current_user.query_count >= 100:
            abort(403, 'Query limit reached for the Free Tier')
        elif current_user.tier == 'pro' and current_user.query_count >= 1000:
            abort(403, 'Query limit reached for the Pro Tier')

@app.route('/')
def index():
    feature_flags = FEATURE_FLAGS.get(current_user.tier, []) if current_user.is_authenticated else []
    current_user_data = None
    if current_user.is_authenticated:
        current_user_data = {
            "username": current_user.username,
            "tier": current_user.tier
        }
    return render_template('index.html', feature_flags=feature_flags, current_user=current_user_data)

@app.route('/features_and_pricing')
def features_and_pricing():
    return render_template('features_and_pricing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        tier = 'free'

        if User.query.filter_by(email=email).first():
            flash('Email address already exists')
            return redirect(url_for('register'))

        new_user = User(
            username=username, 
            email=email, 
            password=generate_password_hash(password, method='pbkdf2:sha256'), 
            tier=tier, 
            query_reset_date=datetime.now(timezone.utc) + timedelta(days=30)
        )
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))

        login_user(user)
        return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/meta_llm')
def meta_llm():
    return render_template('meta_llm.html')

@app.route('/xllm6')
def xllm6():
    if current_user.is_authenticated:
        current_user_data = {
            "is_authenticated": True,
            "username": current_user.username,
            "tier": current_user.tier
        }
    else:
        current_user_data = {"is_authenticated": False}

    feature_flags = FEATURE_FLAGS.get(current_user.tier, []) if current_user.is_authenticated else []

    return render_template('xllm6.html', current_user=current_user_data, feature_flags=feature_flags)

@app.route('/get_responses', methods=['POST'])
def get_responses():
    prompts = request.json.get('prompts')
    responses = asyncio.run(meta_llm_batch(prompts))
    return jsonify(responses)

@app.route('/xllm6_process', methods=['POST'])
def xllm6_process():
    query = request.json.get('query')
    response = llm6_process(query)
    return jsonify(response)

def llm6_process(query):
    tokens = query.lower().split()
    relevant_words = [word for word in tokens if word in dictionary]

    if not relevant_words:
        return {"message": "No relevant words found in the dictionary."}

    result = []
    for word in relevant_words:
        word_info = {
            "word": word,
            "frequency": dictionary.get(word, 0),
            "embedding": embeddings.get(word, {})
        }
        result.append(word_info)

    visualize_embeddings({word: embeddings[word] for word in relevant_words if word in embeddings})
    cluster_data = cluster_embeddings({word: embeddings[word] for word in relevant_words if word in embeddings})

    return {
        "results": result,
        "clusters": cluster_data
    }

dictionary = {}
embeddings = {}

@app.route('/initialize_data', methods=['POST'])
def initialize_data():
    global dictionary
    global embeddings

    # Load dictionary and embeddings from JSON files
    with open('dictionary.json', 'r') as f:
        dictionary = json.load(f)
    
    with open('embeddings.json', 'r') as f:
        embeddings = json.load(f)
    
    logging.info('Dictionary and embeddings initialized.')
    logging.info(f'Dictionary keys: {list(dictionary.keys())[:10]}')  # Log first 10 keys
    logging.info(f'Embeddings keys: {list(embeddings.keys())[:10]}')  # Log first 10 keys
    
    return jsonify({"message": "Data initialized successfully."}), 200

@app.route('/search', methods=['POST'])
@login_required
def search():
    data = request.json
    query = data.get('query')

    if not query:
        logging.error('No query provided in the request.')
        return jsonify({'error': 'No query provided'}), 400

    logging.info(f'Search query received: {query}')

    if not feature_enabled('basic_search'):
        logging.error(f'User {current_user.username} does not have access to the basic_search feature.')
        abort(403, 'Feature not available in your tier')

    current_user.query_count += 1
    db.session.commit()

    results = llm6.semantic_search(query, dictionary)
    logging.info(f'Search results: {results}')
    
    embeddings_result = {}
    for key in results:
        for term, score in results[key]:
            if term in embeddings:
                embeddings_result[term] = embeddings[term]
            else:
                for embed_key in embeddings.keys():
                    if term in embed_key:
                        embeddings_result[embed_key] = embeddings[embed_key]

    logging.info(f'Embeddings result: {list(embeddings_result.keys())}')  # Log the found embeddings keys

    if not embeddings_result:
        logging.error(f'No relevant embeddings found for query: {query}')
        return jsonify({
            'results': results,
            'error': 'No relevant embeddings found'
        }), 200

    clusters_result = cluster_embeddings(embeddings_result)

    response = {
        'results': results,
        'embeddings': prepare_embeddings_for_visualization(embeddings_result),
        'clusters': prepare_clusters_for_visualization(clusters_result),
        'wordCloud': prepare_word_cloud(embeddings_result),
        'distribution': prepare_distribution(embeddings_result)
    }
    return jsonify(response)


@app.route('/export', methods=['GET'])
@login_required
def export():
    if not feature_enabled('data_export'):
        abort(403, 'Feature not available in your tier')

    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Word', 'Frequency'])
    for word, freq in dictionary.items():
        writer.writerow([word, freq])

    output.seek(0)
    return output.getvalue(), 200, {
        'Content-Disposition': 'attachment; filename="results.csv"',
        'Content-Type': 'text/csv'
    }

@app.route('/advanced_feature')
@login_required
@tier_required('pro')
def advanced_feature():
    return render_template('advanced_feature.html')

@app.route('/enterprise_feature')
@login_required
@tier_required('enterprise')
def enterprise_feature():
    return render_template('enterprise_feature.html')

def prepare_embeddings_for_visualization(embeddings):
    words = list(embeddings.keys())
    vectors = [list(embeddings[word].values()) for word in words]
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)
    return {
        'words': words,
        'x': reduced_vectors[:, 0].tolist(),
        'y': reduced_vectors[:, 1].tolist()
    }

def prepare_clusters_for_visualization(clusters):
    return [{
        'x': [vector[0] for vector in cluster],
        'y': [vector[1] for vector in cluster]
    } for cluster in clusters]

def prepare_word_cloud(embeddings):
    words = list(embeddings.keys())
    sizes = [len(word) for word in words]
    return {
        'words': words,
        'x': list(range(len(words))),
        'y': list(range(len(words))),
        'sizes': sizes
    }

def prepare_distribution(embeddings):
    words = list(embeddings.keys())
    frequencies = [len(word) for word in words]
    return {
        'x': frequencies
    }

def cluster_embeddings(embeddings, n_clusters=5):
    words = list(embeddings.keys())
    vectors = [list(embeddings[word].values()) for word in words]

    if not vectors:
        logging.error('No vectors found for clustering.')
        return []

    max_length = max(len(vector) for vector in vectors)
    vectors = [vector + [0] * (max_length - len(vector)) for vector in vectors]

    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(vectors)
    clustered_data = [[] for _ in range(n_clusters)]

    for word, vector, cluster in zip(words, vectors, clusters):
        clustered_data[cluster].append(vector)

    return clustered_data

def plot_word_cloud(words):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def compute_pca(vectors, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(vectors)
    return reduced_vectors

def generate_clusters(vectors, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(vectors)
    return clusters

@app.route('/pca_plot', methods=['POST'])
@login_required
def pca_plot():
    data = request.json
    words = data.get('words', [])
    vectors = [embeddings[word] for word in words if word in embeddings]

    if len(vectors) < 2:
        return jsonify({'error': 'Not enough data points to perform PCA. Need at least 2.'}), 400

    reduced_vectors = compute_pca(vectors)
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

    plt.title('PCA Plot of Word Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

    return jsonify({'message': 'PCA plot generated successfully.'}), 200

@app.route('/cluster_plot', methods=['POST'])
@login_required
def cluster_plot():
    data = request.json
    words = data.get('words', [])
    vectors = [embeddings[word] for word in words if word in embeddings]

    if len(vectors) < 5:
        return jsonify({'error': 'Not enough data points to perform clustering. Need at least 5.'}), 400

    clusters = generate_clusters(vectors)
    reduced_vectors = compute_pca(vectors)
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters)

    for i, word in enumerate(words):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

    plt.title('Cluster Plot of Word Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

    return jsonify({'message': 'Cluster plot generated successfully.'}), 200

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with app.app_context():
        db.create_all()
    app.run(debug=True)
