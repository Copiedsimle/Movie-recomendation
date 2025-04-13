import numpy as np
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, n_factors: int = 50, learning_rate: float = 0.01, 
                 regularization: float = 0.02, n_epochs: int = 20):
        """
        Initialize the collaborative filtering model.
        
        Args:
            n_factors (int): Number of latent factors
            learning_rate (float): Learning rate for gradient descent
            regularization (float): Regularization parameter
            n_epochs (int): Number of training epochs
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        self.movie_ids = None
        
    def fit(self, ratings_matrix: np.ndarray, movie_ids: np.ndarray) -> None:
        """
        Fit the model to the ratings matrix.
        
        Args:
            ratings_matrix (np.ndarray): User-item ratings matrix
            movie_ids (np.ndarray): Array of movie IDs corresponding to columns in ratings_matrix
        """
        n_users, n_items = ratings_matrix.shape
        self.movie_ids = movie_ids
        
        # Initialize user and item factors with random values
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Training loop
        for epoch in range(self.n_epochs):
            for u in range(n_users):
                for i in range(n_items):
                    if ratings_matrix[u, i] > 0:
                        # Calculate prediction error
                        prediction = np.dot(self.user_factors[u], self.item_factors[i])
                        error = ratings_matrix[u, i] - prediction
                        
                        # Update factors
                        self.user_factors[u] += self.learning_rate * (
                            error * self.item_factors[i] - 
                            self.regularization * self.user_factors[u]
                        )
                        self.item_factors[i] += self.learning_rate * (
                            error * self.user_factors[u] - 
                            self.regularization * self.item_factors[i]
                        )
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id (int): User ID
            item_id (int): Item ID
            
        Returns:
            float: Predicted rating
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Find the index of the movie in our matrix
        item_idx = np.where(self.movie_ids == item_id)[0]
        if len(item_idx) == 0:
            raise ValueError(f"Movie ID {item_id} not found in training data")
            
        return np.dot(self.user_factors[user_id], self.item_factors[item_idx[0]])
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Generate movie recommendations for a user.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to generate
            
        Returns:
            List of tuples containing (movie_id, predicted_rating)
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Calculate predicted ratings for all movies
        user_vector = self.user_factors[user_id]
        predicted_ratings = np.dot(self.item_factors, user_vector)
        
        # Get top n recommendations
        top_indices = np.argsort(predicted_ratings)[-n_recommendations:][::-1]
        recommendations = [
            (self.movie_ids[idx], predicted_ratings[idx]) 
            for idx in top_indices
        ]
        
        return recommendations 