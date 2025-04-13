import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os

class MovieLensDataLoader:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the MovieLens data loader.
        
        Args:
            data_dir (str): Directory where the data files are stored
        """
        self.data_dir = data_dir
        self.movies_df = None
        self.ratings_df = None
        self.users_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load the MovieLens dataset.
        
        Returns:
            Tuple containing:
            - movies_df: DataFrame with movie information
            - ratings_df: DataFrame with user ratings
            - users_df: DataFrame with user information
        """
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Data directory '{self.data_dir}' not found. "
                "Please run download_data.py first."
            )
            
        # Check if required files exist
        movies_path = os.path.join(self.data_dir, "ml-latest-small", "movies.csv")
        ratings_path = os.path.join(self.data_dir, "ml-latest-small", "ratings.csv")
        
        if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
            raise FileNotFoundError(
                "MovieLens dataset files not found. "
                "Please run download_data.py first."
            )
        
        # Load movies data
        self.movies_df = pd.read_csv(movies_path)
        
        # Load ratings data
        self.ratings_df = pd.read_csv(ratings_path)
        
        # Create a simple users dataframe
        self.users_df = pd.DataFrame({
            'userId': self.ratings_df['userId'].unique(),
            'userIndex': range(len(self.ratings_df['userId'].unique()))
        })
        
        return self.movies_df, self.ratings_df, self.users_df
    
    def get_user_ratings_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a user-item ratings matrix.
        
        Returns:
            Tuple containing:
            - ratings_matrix: User-item ratings matrix where rows are users and columns are movies
            - movie_ids: Array of movie IDs corresponding to columns in the matrix
        """
        if self.ratings_df is None or self.movies_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Create a pivot table of user ratings
        ratings_matrix = self.ratings_df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        # Get movie IDs in the same order as the matrix columns
        movie_ids = ratings_matrix.columns.values
        
        return ratings_matrix.values, movie_ids
    
    def get_movie_titles(self) -> pd.Series:
        """
        Get movie titles with their IDs.
        
        Returns:
            pd.Series: Series with movieId as index and titles as values
        """
        if self.movies_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        return self.movies_df.set_index('movieId')['title'] 