import numpy as np
from typing import List, Tuple
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error (RMSE).
    
    Args:
        y_true (np.ndarray): True ratings
        y_pred (np.ndarray): Predicted ratings
        
    Returns:
        float: RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def precision_at_k(recommendations: List[Tuple[int, float]], 
                  relevant_items: List[int], k: int = 10) -> float:
    """
    Calculate Precision@K.
    
    Args:
        recommendations (List[Tuple[int, float]]): List of recommended items with scores
        relevant_items (List[int]): List of relevant items
        k (int): Number of recommendations to consider
        
    Returns:
        float: Precision@K score
    """
    recommended_items = [item[0] for item in recommendations[:k]]
    relevant_recommended = len(set(recommended_items) & set(relevant_items))
    return relevant_recommended / k

def recall_at_k(recommendations: List[Tuple[int, float]], 
                relevant_items: List[int], k: int = 10) -> float:
    """
    Calculate Recall@K.
    
    Args:
        recommendations (List[Tuple[int, float]]): List of recommended items with scores
        relevant_items (List[int]): List of relevant items
        k (int): Number of recommendations to consider
        
    Returns:
        float: Recall@K score
    """
    recommended_items = [item[0] for item in recommendations[:k]]
    relevant_recommended = len(set(recommended_items) & set(relevant_items))
    return relevant_recommended / len(relevant_items) if relevant_items else 0.0

def evaluate_model(model, test_data: np.ndarray, 
                  user_id: int, k: int = 10) -> dict:
    """
    Evaluate the recommendation model.
    
    Args:
        model: Trained recommendation model
        test_data (np.ndarray): Test data matrix
        user_id (int): User ID to evaluate
        k (int): Number of recommendations to consider
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get user's relevant items from test data
    relevant_items = np.where(test_data[user_id] > 0)[0].tolist()
    
    # Get recommendations
    recommendations = model.recommend(user_id, n_recommendations=k)
    
    # Calculate metrics
    metrics = {
        'precision@k': precision_at_k(recommendations, relevant_items, k),
        'recall@k': recall_at_k(recommendations, relevant_items, k)
    }
    
    return metrics 