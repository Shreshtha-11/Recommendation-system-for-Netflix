import numpy as np
import pandas as pd

def calculate_rmse(y_true, y_pred):
    """
    Computes Root Mean Squared Error (RMSE) between actual and predicted ratings.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mae(y_true, y_pred):
    """
    Computes Mean Absolute Error (MAE) between actual and predicted ratings.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def calculate_ap_at_k(recommended_items, actual_relevant_items, k=10):
    """
    Computes Average Precision at K (AP@K) for a single user.
    """
    if not actual_relevant_items:
        return 0.0
        
    recommended_items = list(recommended_items)[:k]
    score = 0.0
    num_hits = 0.0
    
    for i, item in enumerate(recommended_items):
        if item in actual_relevant_items:
            num_hits += 1.0
            precision_at_i = num_hits / (i + 1.0)
            score += precision_at_i
            
    denominator = min(k, len(actual_relevant_items))
    return score / denominator if denominator > 0 else 0.0

def calculate_map_at_k(all_recommendations, test_df, k=10, relevance_threshold=3.5):
    """
    Computes Mean Average Precision at K (MAP@K) across test users who have relevant items.
    """
    # Filter test_df to relevant items only
    relevant_test_df = test_df[test_df["Rating"] >= relevance_threshold]
    
    # Group relevant items by user
    user_relevant_items = relevant_test_df.groupby("UserID")["MovieID"].apply(set).to_dict()
    
    ap_scores = []
    # Evaluate MAP only for users who have at least one relevant item in the test set
    for user_id in user_relevant_items.keys():
        recs = all_recommendations.get(user_id, [])
        actual_rel = user_relevant_items[user_id]
        
        ap = calculate_ap_at_k(recs, actual_rel, k=k)
        ap_scores.append(ap)
        
    return np.mean(ap_scores) if ap_scores else 0.0

def calculate_precision_recall_at_k(all_recommendations, test_df, k=10, relevance_threshold=3.5):
    """
    Computes Precision@K and Recall@K across all test users who have relevant items.
    """
    relevant_test_df = test_df[test_df["Rating"] >= relevance_threshold]
    user_relevant_items = relevant_test_df.groupby("UserID")["MovieID"].apply(set).to_dict()
    
    precisions = []
    recalls = []
    
    for user_id in user_relevant_items.keys():
        recs = list(all_recommendations.get(user_id, []))[:k]
        actual_rel = user_relevant_items[user_id]
        
        n_rel = len(actual_rel)
        n_rec = len(recs)
        
        n_rel_and_rec = len([item for item in recs if item in actual_rel])
        
        precision = n_rel_and_rec / n_rec if n_rec > 0 else 0.0
        recall = n_rel_and_rec / n_rel if n_rel > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        
    return np.mean(precisions) if precisions else 0.0, np.mean(recalls) if recalls else 0.0
