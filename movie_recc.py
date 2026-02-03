from datetime import date
import os
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, redirect, request, render_template, session
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
import requests

#variables being used 
cv = TfidfVectorizer() #CountVectorizer()
movie_data = pd.read_csv('movie.csv')

# Ensure a `year` column exists — derive from `release_date` when missing.
if 'year' not in movie_data.columns:
    if 'release_date' in movie_data.columns:
        movie_data['release_date'] = pd.to_datetime(movie_data['release_date'], errors='coerce')
        movie_data['year'] = movie_data['release_date'].dt.year
    else:
        movie_data['year'] = np.nan
    # Fill missing years with 1900 (keeps recency score conservative) and make int
    movie_data['year'] = movie_data['year'].fillna(1900).astype(int)

recommendation_cache = {} # a dictionary
#processing 
movie_features = movie_data[['keywords','cast','genres','director', 'tagline']]
#merging into one column 
movie_data['combined'] = movie_data.apply(lambda row: ' '.join(row.astype(str)), axis=1)

#get the count and cosine sim
matrix = cv.fit_transform(movie_data['combined'])
similarity = cosine_similarity(matrix)


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
    current_year = date.today().year
    # Determine selected movie year safely (fallback to current year)
    if 'year' in movie_data_copy.columns:
        try:
            selected_movie_year = int(movie_data_copy.at[movie_index, 'year'])
        except Exception:
            selected_movie_year = current_year
    else:
        # try to derive from release_date if available
        if 'release_date' in movie_data_copy.columns:
            try:
                sel_date = pd.to_datetime(movie_data_copy.at[movie_index, 'release_date'], errors='coerce')
                selected_movie_year = int(sel_date.year) if not pd.isna(sel_date) else current_year
            except Exception:
                selected_movie_year = current_year
        else:
            selected_movie_year = current_year

    # Ensure a numeric `year` column exists for the whole dataframe
    if 'year' not in movie_data_copy.columns:
        if 'release_date' in movie_data_copy.columns:
            movie_data_copy['release_date'] = pd.to_datetime(movie_data_copy['release_date'], errors='coerce')
            movie_data_copy['year'] = movie_data_copy['release_date'].dt.year.fillna(1900).astype(int)
        else:
            movie_data_copy['year'] = 1900

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
    if title not in session['history']:
        session['history'].append(title)

def get_movie_poster(title):
    api_key = '4aeebfc8a7f0cf841fd70b3f9288c5db'
    url = "https://api.themoviedb.org/3/search/movie"
    params = {'api_key': api_key, 'query': title}
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return ""

    results = data.get('results', [])
    if not results:
        return ""

    poster_path = results[0].get('poster_path')
    if not poster_path:
        return ""

    return f"https://image.tmdb.org/t/p/w500{poster_path}"

#getting input from user in (back-end)
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Disable caching for static files during development
@app.after_request
def set_cache_headers(response):
    response.cache_control.max_age = 0
    response.cache_control.no_cache = True
    response.cache_control.no_store = True
    response.cache_control.must_revalidate = True
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    user_movie = request.form.get('user_movie')
    user_movie_copy = user_movie.strip().lower()
    
    
    if user_movie_copy in recommendation_cache:
        posters = recommendation_cache[user_movie_copy]
        return render_template('index.html', movie=user_movie, recommended=posters,found=1)
    else:
        title_match = find_movie(user_movie_copy)
        if title_match is None or title_match[1]<75:
            return render_template('index.html', movie=user_movie, recommended=["Movie not found."],found=0)
        else:
            add_to_history(user_movie)
            movie_index = title_match[-1]
        
            recommended_df = get_recommended(movie_index).tolist()
            recommended = recommended_df
            if title_match[0] in recommended:
                recommended.remove(title_match[0])
            
            posters=[]
            for movie in recommended: 
                poster_url = get_movie_poster(movie)
                posters.append((movie, poster_url))
            
            recommendation_cache[user_movie_copy] = posters
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
    # Run on a non-default port to avoid macOS services occupying port 5000.
    # Disable the auto-reloader so running via terminal blocks and shows logs.
    app.run(debug=True, port=5001, use_reloader=False)

