import os
import pandas as pd 
import numpy as np 
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, redirect, request, render_template, session
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
import requests

#variables being used 
try:
    movie_data = pd.read_csv('movie.csv')
    print(f"✓ Loaded {len(movie_data)} movies from movie.csv")
except FileNotFoundError:
    print("✗ Error: movie.csv not found. Make sure it's in the same directory as movie_recc.py")
    exit(1)
except Exception as e:
    print(f"✗ Error loading movie.csv: {e}")
    exit(1)

# Data preprocessing
movie_data['release_date'] = pd.to_datetime(movie_data['release_date'], errors='coerce')
movie_data['year'] = movie_data['release_date'].dt.year
movie_data['runtime'] = pd.to_numeric(movie_data['runtime'], errors='coerce').fillna(movie_data['runtime'].median())

# Fill missing values to prevent vectorizer issues
movie_data['genres'] = movie_data['genres'].fillna('')
movie_data['keywords'] = movie_data['keywords'].fillna('')
movie_data['cast'] = movie_data['cast'].fillna('')
movie_data['director'] = movie_data['director'].fillna('')

recommendation_cache = {} 

# Create weighted feature matrix for better similarity matching
# Separate vectorizers for different features
try:
    print("Building similarity matrices...")
    genre_vec = TfidfVectorizer(max_features=100, analyzer='char', ngram_range=(2,2))
    keyword_vec = TfidfVectorizer(max_features=100)
    cast_vec = TfidfVectorizer(max_features=50, analyzer='char', ngram_range=(2,2))
    director_vec = TfidfVectorizer(max_features=50, analyzer='char', ngram_range=(2,2))

    # Fit vectorizers
    genre_matrix = genre_vec.fit_transform(movie_data['genres'])
    keyword_matrix = keyword_vec.fit_transform(movie_data['keywords'])
    cast_matrix = cast_vec.fit_transform(movie_data['cast'])
    director_matrix = director_vec.fit_transform(movie_data['director'])

    # Calculate individual similarities (40% genres, 30% keywords, 20% cast, 10% director)
    genre_similarity = cosine_similarity(genre_matrix)
    keyword_similarity = cosine_similarity(keyword_matrix)
    cast_similarity = cosine_similarity(cast_matrix)
    director_similarity = cosine_similarity(director_matrix)

    # Weighted combined similarity
    similarity = (0.4 * genre_similarity + 
                  0.3 * keyword_similarity + 
                  0.2 * cast_similarity + 
                  0.1 * director_similarity)
    
    print("✓ Similarity matrices built successfully")
except Exception as e:
    print(f"✗ Error building similarity matrices: {e}")
    exit(1)


#functions for processing

mean_vote_data = movie_data['vote_average'].mean()
min_votes = movie_data['vote_count'].quantile(0.8)

def weighted_rating(x, mean_vote_data=mean_vote_data, min_votes=min_votes):
    num_votes = x['vote_count']
    avg_rating = x['vote_average']
    
    return (num_votes/(num_votes+min_votes))*avg_rating + (min_votes/(num_votes+min_votes))*mean_vote_data 

def get_recommended(movie_index):
    """
    Improved recommendation system with better feature weighting and filtering.
    Balances content-based similarity with quality metrics while avoiding popularity bias.
    """
    # Weights for different components
    similarity_weight = 0.65  # Content-based similarity is primary
    quality_weight = 0.25    # Rating quality
    recency_weight = 0.10    # Slight preference for reasonably recent content
    
    movie_data_copy = movie_data.copy()
    
    # 1. Calculate quality score (normalized weighted rating)
    movie_data_copy['quality_score'] = movie_data_copy.apply(weighted_rating, axis=1)
    
    # 2. Calculate recency score (normalized by year difference)
    current_year = datetime.now().year
    selected_movie_year = movie_data_copy.loc[movie_index, 'year']
    movie_data_copy['year_diff'] = abs(movie_data_copy['year'] - selected_movie_year)
    movie_data_copy['recency_score'] = 1 - (movie_data_copy['year_diff'] / (current_year - 1900))
    movie_data_copy['recency_score'] = movie_data_copy['recency_score'].clip(0, 1)
    
    # 3. Normalize scores to 0-1 range
    scaler = MinMaxScaler()
    
    quality_scores = scaler.fit_transform(movie_data_copy[['quality_score']])
    movie_data_copy['quality_score_norm'] = quality_scores
    
    recency_scores = scaler.fit_transform(movie_data_copy[['recency_score']])
    movie_data_copy['recency_score_norm'] = recency_scores
    
    # 4. Add similarity scores from content features
    movie_data_copy['similarity_score'] = similarity[movie_index]
    similarity_norm = scaler.fit_transform(movie_data_copy[['similarity_score']])
    movie_data_copy['similarity_norm'] = similarity_norm
    
    # 5. Calculate final hybrid score
    movie_data_copy['hybrid_score'] = (
        similarity_weight * movie_data_copy['similarity_norm'] +
        quality_weight * movie_data_copy['quality_score_norm'] +
        recency_weight * movie_data_copy['recency_score_norm']
    )
    
    # 6. Filter out the selected movie and poor matches
    recommended = movie_data_copy[
        (movie_data_copy.index != movie_index) & 
        (movie_data_copy['similarity_score'] > 0.1)  # Only movies with meaningful similarity
    ].sort_values('hybrid_score', ascending=False)
    
    return recommended['title'].head(10)

def find_movie(title):
    movie_data_copy = movie_data.copy()
    movie_data_copy['title']=movie_data_copy['title'].str.lower().str.strip()
    choices = list(zip(movie_data_copy['title'],movie_data_copy.index))

    return process.extractOne(title,choices,scorer=fuzz.WRatio)

def add_to_history(title):
    if 'history' not in session:
        session['history'] = []
    session['history'].append(title)

def get_movie_poster(title):
    api_key = '4aeebfc8a7f0cf841fd70b3f9288c5db'
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}"
    try:
        response = requests.get(url, timeout=5).json()
        results = response.get('results')
        if not results: 
            return ""
        
        poster_path = results[0].get('poster_path')
        if not poster_path: 
            return ""
        full_poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        return full_poster_url
    except Exception as e:
        print(f"Error fetching poster for {title}: {e}")
        return ""

#getting input from user in (back-end)
app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    user_movie = request.form.get('user_movie')
    user_movie_copy = user_movie.strip().lower()
    
    
    if user_movie_copy in recommendation_cache:
        recommended = recommendation_cache[user_movie_copy]
        return render_template('index.html', movie=user_movie, recommended=recommended,found=1)
    else:
        title_match = find_movie(user_movie_copy)
        if title_match==None or title_match[1]<75:
            return render_template('index.html', movie=user_movie, recommended=["Movie not found."],found=0)
        else:
            add_to_history(user_movie)
            movie_index = title_match[-1]
        
            recommended_df = get_recommended(movie_index).tolist()
            recommended = recommended_df
            if title_match[0] in recommended:
                recommended.remove(title_match[0])
            
            recommendation_cache[user_movie_copy] = recommended
            posters=[]
            for movie in recommended: 
                poster_url = get_movie_poster(movie)
                posters.append((movie, poster_url))
            return render_template('index.html', movie=user_movie, recommended=posters,found=1)

@app.route('/history', methods=['GET'])
def history():
    user_history = session.get('history',[])
    return render_template('index.html', history=user_history)

@app.route('/delete_history', methods=['POST'])
def delete_history():
    session['history'] = []
    session.modified = True
    return redirect('/')

if __name__=='__main__':
    print("\n🎬 Movie Recommendation Engine Starting...")
    print(f"📊 Dataset: {len(movie_data)} movies")
    print("🌐 Server running at http://127.0.0.1:5000/")
    print("Press CTRL+C to stop.\n")
    app.run(debug=True)

