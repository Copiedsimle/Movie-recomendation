import numpy as np
from typing import List, Tuple

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
        
    def fit(self, ratings_matrix: np.ndarray, movie_ids: List[int]) -> None:
        """Train the model using matrix factorization"""
        n_users, n_items = ratings_matrix.shape
        
        # Initialize factors randomly
        if self.user_factors is None or self.item_factors is None:
            self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
            self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # Store movie IDs for later use
        self.movie_ids = movie_ids
        
        # Training loop
        for epoch in range(self.n_epochs):
            for u in range(n_users):
                for i in range(n_items):
                    if ratings_matrix[u, i] > 0:  # Only train on observed ratings
                        # Compute prediction and error
                        prediction = np.dot(self.user_factors[u], self.item_factors[i])
                        error = ratings_matrix[u, i] - prediction
                        
                        # Update factors
                        self.user_factors[u] += self.learning_rate * (error * self.item_factors[i] - self.regularization * self.user_factors[u])
                        self.item_factors[i] += self.learning_rate * (error * self.user_factors[u] - self.regularization * self.item_factors[i])

    def recommend(self, user_id: int, n_recommendations: int = 5) -> List[Tuple[int, float]]:
        """Generate movie recommendations for a user"""
        if self.user_factors is None or self.item_factors is None:
            return []
        
        try:
            # Get all predicted ratings for the user
            user_ratings = np.dot(self.user_factors[user_id], self.item_factors.T)
            
            # Create a list of (movie_index, predicted_rating) tuples
            movie_ratings = [(i, rating) for i, rating in enumerate(user_ratings)]
            
            # Sort by predicted rating in descending order
            movie_ratings.sort(key=lambda x: x[1], reverse=True)
            
            # Return top N recommendations
            return movie_ratings[:n_recommendations]
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for a specific user-item pair"""
        if self.user_factors is None or self.item_factors is None:
            return 0.0
        
        try:
            prediction = np.dot(self.user_factors[user_id], self.item_factors[item_id])
            return max(0.5, min(5.0, prediction))  # Clip prediction between 0.5 and 5.0
        except Exception as e:
            print(f"Error predicting rating: {e}")
            return 0.0 