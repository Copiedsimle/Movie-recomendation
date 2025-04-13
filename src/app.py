from flask import Flask, render_template, request, redirect, url_for, flash
from data.data_loader import MovieLensDataLoader
from models.collaborative_filtering import CollaborativeFiltering
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Initialize data loader and model
data_loader = MovieLensDataLoader()
movies_df, ratings_df, users_df = data_loader.load_data()
ratings_matrix, movie_ids = data_loader.get_user_ratings_matrix()
movie_titles = data_loader.get_movie_titles()

# Create a mapping between movie IDs and matrix indices
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
idx_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}

# Initialize the model
model = CollaborativeFiltering(n_factors=50, learning_rate=0.01, 
                             regularization=0.02, n_epochs=20)
model.fit(ratings_matrix, movie_ids)

@app.route('/')
def index():
    # Get popular movies (movies with highest average ratings)
    popular_movies = []
    for movie_id in movie_ids:
        idx = movie_id_to_idx[movie_id]
        ratings = ratings_matrix[:, idx]
        if np.sum(ratings > 0) > 0:  # Only consider movies with ratings
            avg_rating = np.mean(ratings[ratings > 0])
            popular_movies.append({
                'id': movie_id,
                'title': movie_titles.get(movie_id, f"Movie {movie_id}"),
                'rating': avg_rating
            })
    
    # Sort by rating and take top 12
    popular_movies.sort(key=lambda x: x['rating'], reverse=True)
    popular_movies = popular_movies[:12]
    
    return render_template('index.html', popular_movies=popular_movies)

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '').lower()
    if not query:
        flash('Please enter a search term', 'warning')
        return redirect(url_for('index'))
    
    # Search for movies
    results = []
    for movie_id, title in movie_titles.items():
        if query in title.lower() and movie_id in movie_id_to_idx:
            idx = movie_id_to_idx[movie_id]
            ratings = ratings_matrix[:, idx]
            avg_rating = np.mean(ratings[ratings > 0]) if np.sum(ratings > 0) > 0 else 0
            results.append({
                'id': movie_id,
                'title': title,
                'rating': avg_rating
            })
    
    return render_template('index.html', search_results=results)

@app.route('/rate', methods=['POST'])
def rate_movie():
    movie_id = int(request.form.get('movie_id'))
    rating = int(request.form.get('rating'))
    
    if movie_id not in movie_id_to_idx:
        flash('Invalid movie ID', 'error')
        return redirect(url_for('index'))
    
    # In a real application, you would save this rating to a database
    # For now, we'll just update the ratings matrix
    user_id = 1  # Example user ID
    idx = movie_id_to_idx[movie_id]
    ratings_matrix[user_id, idx] = rating
    
    # Retrain the model with the new rating
    model.fit(ratings_matrix, movie_ids)
    
    flash('Rating submitted successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/recommendations')
def recommendations():
    user_id = 1  # Example user ID
    recommendations = model.recommend(user_id, n_recommendations=10)
    
    recommended_movies = []
    for idx, predicted_rating in recommendations:
        movie_id = idx_to_movie_id[idx]
        recommended_movies.append({
            'id': movie_id,
            'title': movie_titles.get(movie_id, f"Movie {movie_id}"),
            'predicted_rating': predicted_rating
        })
    
    return render_template('recommendations.html', recommendations=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True) 