from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from data.data_loader import MovieLensDataLoader
from models.collaborative_filtering import CollaborativeFiltering
from utils.imdb_integration import IMDbIntegration
import numpy as np
from datetime import datetime, timedelta
from models.rating import Rating
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from functools import wraps
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configure caching
cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Initialize data loader and model
data_loader = MovieLensDataLoader()
movies_df, ratings_df, users_df = data_loader.load_data()
ratings_matrix, movie_ids = data_loader.get_user_ratings_matrix()
movie_titles = data_loader.get_movie_titles()

# Create a mapping between movie IDs and matrix indices
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
idx_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}

# Initialize the models
model = CollaborativeFiltering(n_factors=50, learning_rate=0.01, 
                             regularization=0.02, n_epochs=20)
model.fit(ratings_matrix, movie_ids)

# Initialize IMDb integration
imdb_client = IMDbIntegration()

# Track when the model was last trained
last_model_training = datetime.now()
pending_ratings = []  # Store ratings that haven't been used to retrain the model

# Initialize SQLAlchemy
db = SQLAlchemy()

def async_task(f):
    """Decorator to run a function asynchronously"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        def task():
            try:
                f(*args, **kwargs)
            except Exception as e:
                print(f"Error in async task: {e}")
        from threading import Thread
        Thread(target=task).start()
    return wrapped

@async_task
def update_model_if_needed():
    """Update the model if enough time has passed or we have enough pending ratings"""
    global last_model_training, pending_ratings
    
    current_time = datetime.now()
    time_since_last_training = current_time - last_model_training
    
    # Retrain if it's been more than 1 hour or we have more than 10 pending ratings
    if time_since_last_training > timedelta(hours=1) or len(pending_ratings) >= 10:
        model.fit(ratings_matrix, movie_ids)
        last_model_training = current_time
        pending_ratings = []  # Clear pending ratings
        cache.delete_memoized(get_popular_movies)
        cache.delete_memoized(get_recommendations)

@cache.memoize(timeout=300)
def get_popular_movies(max_results=12):
    """Get popular movies with caching"""
    popular_movies = imdb_client.get_popular_movies(max_results=max_results)
    
    # Get local ratings for these movies if available
    for movie in popular_movies:
        if movie['imdb_id'] in movie_id_to_idx:
            idx = movie_id_to_idx[movie['imdb_id']]
            ratings = ratings_matrix[:, idx]
            if np.sum(ratings > 0) > 0:
                movie['local_rating'] = float(np.mean(ratings[ratings > 0]))
            else:
                movie['local_rating'] = None
        else:
            movie['local_rating'] = None
    
    return popular_movies

@cache.memoize(timeout=300)
def get_recommendations(user_id, n_recommendations=10):
    """Get recommendations with caching"""
    try:
        movie_recommendations = model.recommend(user_id, n_recommendations=n_recommendations)
        
        if not movie_recommendations:
            return []
        
        recommended_movies = []
        for movie_idx, predicted_rating in movie_recommendations:
            try:
                imdb_id = idx_to_movie_id.get(int(movie_idx))
                if imdb_id:
                    movie_details = imdb_client.get_movie_details(imdb_id)
                    if movie_details:
                        movie_details['predicted_rating'] = round(float(predicted_rating), 1)
                        recommended_movies.append(movie_details)
            except Exception as e:
                print(f"Error processing movie {movie_idx}: {e}")
                continue
                
        return recommended_movies
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return []

@app.route('/')
def index():
    # Get popular movies (now cached)
    popular_movies = get_popular_movies(max_results=48)  # Increased for pagination
    return render_template('index.html', 
                         popular_movies=popular_movies,
                         recommended_movies=[] if not hasattr(request, 'user') else get_recommendations(request.user.id))

@app.route('/search')
@cache.memoize(timeout=60)
def search():
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({'results': []})
    
    # Search using IMDb (now cached)
    results = imdb_client.search_movie(query)
    
    # Get local ratings for these movies if available
    for movie in results:
        if movie['imdb_id'] in movie_id_to_idx:
            idx = movie_id_to_idx[movie['imdb_id']]
            ratings = ratings_matrix[:, idx]
            movie['local_rating'] = float(np.mean(ratings[ratings > 0])) if np.sum(ratings > 0) > 0 else None
        else:
            movie['local_rating'] = None
    
    return jsonify({'results': results})

@app.route('/movie/<imdb_id>')
@cache.memoize(timeout=300)
def movie_details(imdb_id):
    # Get movie details from IMDb (now cached)
    details = imdb_client.get_movie_details(imdb_id)
    if not details:
        flash('Movie not found', 'error')
        return redirect(url_for('index'))
    
    # Get local rating if available
    if imdb_id in movie_id_to_idx:
        idx = movie_id_to_idx[imdb_id]
        ratings = ratings_matrix[:, idx]
        details['local_rating'] = float(np.mean(ratings[ratings > 0])) if np.sum(ratings > 0) > 0 else None
    else:
        details['local_rating'] = None
    
    # Get similar movies (now cached)
    similar_movies = imdb_client.get_movie_recommendations(imdb_id)
    
    return render_template('movie_details.html', movie=details, similar_movies=similar_movies)

@app.route('/rate_movie', methods=['POST'])
def rate_movie():
    try:
        user_id = int(request.form['user_id'])
        imdb_id = request.form['imdb_id']
        rating_value = float(request.form['rating'])
        
        if not (0.5 <= rating_value <= 5.0):
            return jsonify({'success': False, 'message': 'Invalid rating value'})
            
        # Check if rating exists
        rating = Rating.query.filter_by(
            user_id=user_id,
            imdb_id=imdb_id
        ).first()
        
        if rating:
            # Update existing rating
            rating.rating = rating_value
            rating.timestamp = datetime.utcnow()
        else:
            # Create new rating
            rating = Rating(
                user_id=user_id,
                imdb_id=imdb_id,
                rating=rating_value,
                timestamp=datetime.utcnow()
            )
            db.session.add(rating)
            
        db.session.commit()
        
        # Update the model asynchronously
        update_model_if_needed()
        
        # Clear relevant caches
        cache.delete_memoized(get_recommendations, user_id)
        
        return jsonify({'success': True, 'message': 'Rating submitted successfully'})
        
    except Exception as e:
        print(f"Error submitting rating: {e}")
        return jsonify({'success': False, 'message': 'Error submitting rating'})

@app.route('/recommendations/<int:user_id>')
def recommendations(user_id):
    try:
        recommended_movies = get_recommendations(user_id)
        
        if not recommended_movies:
            return jsonify({'success': False, 'message': 'No recommendations available'})
            
        return jsonify({
            'success': True,
            'recommendations': recommended_movies
        })
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return jsonify({'success': False, 'message': 'Error generating recommendations'})

@async_task
def update_model():
    """Update the collaborative filtering model with current ratings"""
    try:
        # Get all ratings from the database
        ratings = Rating.query.all()
        
        if not ratings:
            print("No ratings available to train the model")
            return
            
        # Create user-movie rating matrix
        users = sorted(list(set(r.user_id for r in ratings)))
        movies = sorted(list(set(r.imdb_id for r in ratings)))
        
        # Create mappings
        global user_to_idx, movie_to_idx, idx_to_movie_id
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        movie_to_idx = {movie: idx for idx, movie in enumerate(movies)}
        idx_to_movie_id = {idx: movie for movie, idx in movie_to_idx.items()}
        
        # Initialize rating matrix
        rating_matrix = np.zeros((len(users), len(movies)))
        
        # Fill rating matrix
        for rating in ratings:
            user_idx = user_to_idx[rating.user_id]
            movie_idx = movie_to_idx[rating.imdb_id]
            rating_matrix[user_idx, movie_idx] = rating.rating
            
        # Train the model
        model.fit(rating_matrix, movies)
        print("Model updated successfully")
        
        # Clear all relevant caches
        cache.delete_memoized(get_popular_movies)
        cache.delete_memoized(get_recommendations)
        
    except Exception as e:
        print(f"Error updating model: {e}")

if __name__ == '__main__':
    app.run(debug=True) 