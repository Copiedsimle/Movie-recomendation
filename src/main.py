import numpy as np
from data.data_loader import MovieLensDataLoader
from data.download_data import download_movielens_data
from models.collaborative_filtering import CollaborativeFiltering
from utils.evaluation import evaluate_model

def main():
    # Download the dataset first
    print("Downloading MovieLens dataset...")
    download_movielens_data()
    
    # Initialize data loader
    data_loader = MovieLensDataLoader()
    
    # Load data
    print("Loading data...")
    movies_df, ratings_df, users_df = data_loader.load_data()
    
    # Get ratings matrix and movie IDs
    print("Creating ratings matrix...")
    ratings_matrix, movie_ids = data_loader.get_user_ratings_matrix()
    
    # Split data into train and test sets
    np.random.seed(42)
    mask = np.random.rand(*ratings_matrix.shape) < 0.8
    train_data = ratings_matrix.copy()
    train_data[~mask] = 0
    test_data = ratings_matrix.copy()
    test_data[mask] = 0
    
    # Initialize and train the model
    print("Training model...")
    model = CollaborativeFiltering(n_factors=50, learning_rate=0.01, 
                                 regularization=0.02, n_epochs=20)
    model.fit(train_data, movie_ids)
    
    # Get movie titles
    movie_titles = data_loader.get_movie_titles()
    
    # Example: Get recommendations for a user
    user_id = 1  # Example user ID
    print(f"\nGenerating recommendations for user {user_id}...")
    recommendations = model.recommend(user_id, n_recommendations=10)
    
    print("\nTop 10 movie recommendations:")
    for movie_id, predicted_rating in recommendations:
        try:
            title = movie_titles[movie_id]
            print(f"{title}: {predicted_rating:.2f}")
        except KeyError:
            print(f"Movie ID {movie_id} not found in database")
    
    # Evaluate the model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, test_data, user_id)
    print(f"Precision@10: {metrics['precision@k']:.4f}")
    print(f"Recall@10: {metrics['recall@k']:.4f}")

if __name__ == "__main__":
    main() 