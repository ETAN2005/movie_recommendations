import os
import pandas as pd 
import numpy as np 
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, redirect, request, render_template, session
from fuzzywuzzy import fuzz
from fuzzywuzzy import process 
import requests

#variables being used 
cv = TfidfVectorizer() #CountVectorizer()
movie_data = pd.read_csv('movie.csv')

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
    alpha = 0.7 #weight for similarity 
    beta = 0.3  #weight for popularity 

    movie_data_copy = movie_data.copy()
    movie_data_copy['popularity_score'] = movie_data_copy.apply(weighted_rating, axis=1)

    if movie_data_copy['popularity_score'].max() != movie_data_copy['popularity_score'].min():
        movie_data_copy['popularity_score'] = (movie_data_copy['popularity_score']-movie_data_copy['popularity_score'].min())/(movie_data_copy['popularity_score'].max()-movie_data_copy['popularity_score'].min())
    else:
        movie_data_copy['popularity_score'] = 0

    movie_data_copy['similarity'] = similarity[movie_index]
    if movie_data_copy['similarity'].max() != movie_data_copy['similarity'].min():
        movie_data_copy['similarity'] = (movie_data_copy['similarity']-movie_data_copy['similarity'].min())/(movie_data_copy['similarity'].max()-movie_data_copy['similarity'].min())
    else:
        movie_data_copy['similarity']=0

    
    #hybrid score 
    movie_data_copy['hybrid'] = alpha * movie_data_copy['similarity'] + beta * movie_data_copy['popularity_score']

    recommended = movie_data_copy[movie_data_copy.index != movie_index].sort_values('hybrid',ascending=False)
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
    response = requests.get(url).json()
    results = response.get('results')
    if not results: 
        return ""
    
    poster_path = results[0].get('results')
    if not poster_path: 
        return ""
    full_poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
    return full_poster_url

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
    session.pop('history', None)
    return redirect('/')

if __name__=='__main__':
    app.run(debug=True)

