import os
import time
import pandas as pd
import numpy as np

from src.data_utils import load_movie_titles, split_data_chronologically, get_user_interactions
from src.metrics import calculate_rmse, calculate_mae, calculate_map_at_k, calculate_precision_recall_at_k
from src.models.base import BaseRecommender
from src.models.collaborative_filtering import UserCollaborativeFiltering, ItemCollaborativeFiltering
from src.models.matrix_factorization import FunkSVD

class BaselineMeanRecommender(BaseRecommender):
    """
    A simple baseline recommender that predicts the user's mean rating if known,
    otherwise the global mean.
    """
    def __init__(self):
        self.global_mean = 3.5
        self.user_means = {}
        
    def fit(self, train_df):
        self.global_mean = train_df["Rating"].mean()
        self.user_means = train_df.groupby("UserID")["Rating"].mean().to_dict()
        
    def predict(self, user_id, movie_id):
        return self.user_means.get(user_id, self.global_mean)

def run_evaluation_pipeline(data_dir="data", max_test_users=100):
    print("="*65)
    print("      NETFLIX RECOMMENDATION SYSTEM PIPELINE      ")
    print("="*65)
    
    movie_titles_path = os.path.join(data_dir, "movie_titles.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")
    
    if not os.path.exists(movie_titles_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Required data files not found in '{data_dir}' folder.")
        
    movies_df = load_movie_titles(movie_titles_path)
    ratings_df = pd.read_csv(ratings_path)
    
    # Cast ratings Date to datetime if not already
    ratings_df["Date"] = pd.to_datetime(ratings_df["Date"])
    
    # Chronological Split
    train_df, test_df = split_data_chronologically(ratings_df, test_ratio=0.2)
    
    all_movie_ids = list(train_df["MovieID"].unique())
    train_interactions = get_user_interactions(train_df)
    
    # Define models
    models = {
        "Baseline Mean": BaselineMeanRecommender(),
        "User-Based CF": UserCollaborativeFiltering(k=20),
        "Item-Based CF": ItemCollaborativeFiltering(k=20),
        "Funk SVD (MF)": FunkSVD(n_factors=15, lr=0.005, reg=0.02, n_epochs=15, patience=3)
    }
    
    results = []
    
    # Train and Evaluate
    for name, model in models.items():
        print("\n" + "-"*50)
        print(f"Model: {name}")
        print("-"*50)
        
        t0 = time.time()
        if name == "Funk SVD (MF)":
            model.fit(train_df, val_df=test_df)
        else:
            model.fit(train_df)
        train_time = time.time() - t0
        print(f"Training completed in {train_time:.2f} seconds.")
        
        # Test predictions
        t0 = time.time()
        test_preds = []
        for _, row in test_df.iterrows():
            test_preds.append(model.predict(row["UserID"], row["MovieID"]))
        prediction_time = time.time() - t0
        
        # Metrics
        rmse = calculate_rmse(test_df["Rating"].values, test_preds)
        mae = calculate_mae(test_df["Rating"].values, test_preds)
        
        # Recommendations (on a sample of test users for speed)
        test_users = test_df["UserID"].unique()
        eval_users = test_users
        if len(test_users) > max_test_users:
            np.random.seed(42)
            eval_users = np.random.choice(test_users, size=max_test_users, replace=False)
            
        print(f"Generating Top-10 recommendations for {len(eval_users)} test users...")
        recs = {}
        for u_id in eval_users:
            watched = train_interactions.get(u_id, set())
            recs[u_id] = model.recommend(u_id, n=10, train_interactions=watched, all_movie_ids=all_movie_ids)
            
        cohort_test_df = test_df[test_df["UserID"].isin(eval_users)]
        map_10 = calculate_map_at_k(recs, cohort_test_df, k=10, relevance_threshold=3.5)
        prec_10, recall_10 = calculate_precision_recall_at_k(recs, cohort_test_df, k=10, relevance_threshold=3.5)
        
        print(f"Results for {name}: RMSE={rmse:.4f} | MAP@10={map_10:.4f}")
        
        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "MAP@10": map_10,
            "Precision@10": prec_10,
            "Recall@10": recall_10,
            "Train Time (s)": train_time
        })
        
    # Print comparison table
    summary_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("                        MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(summary_df.to_string(index=False, formatters={
        "RMSE": "{:.4f}".format,
        "MAE": "{:.4f}".format,
        "MAP@10": "{:.4f}".format,
        "Precision@10": "{:.4f}".format,
        "Recall@10": "{:.4f}".format,
        "Train Time (s)": "{:.2f}".format
    }))
    print("="*80)
    
    # Demo recommendations
    svd_model = models["Funk SVD (MF)"]
    if len(test_users) > 0:
        sample_user = int(np.random.choice(test_users))
        print(f"\nDemo Recommendations for User {sample_user} (using FunkSVD):")
        
        # User history
        user_train = train_df[train_df["UserID"] == sample_user].merge(movies_df, on="MovieID")
        print("\nHistorical interactions in Training Set (Top 5):")
        for _, row in user_train.sort_values(by="Rating", ascending=False).head(5).iterrows():
            print(f"  - {row['Title']} ({row['Year']}) | Rating: {row['Rating']}")
            
        # Actual test set ratings
        user_test = test_df[test_df["UserID"] == sample_user].merge(movies_df, on="MovieID")
        if not user_test.empty:
            print("\nActual items rated in Test Set:")
            for _, row in user_test.iterrows():
                print(f"  - {row['Title']} ({row['Year']}) | Actual Rating: {row['Rating']}")
                
        # Recommended
        watched = train_interactions.get(sample_user, set())
        recs_5 = svd_model.recommend(sample_user, n=5, train_interactions=watched, all_movie_ids=all_movie_ids)
        
        print("\nTop 5 Recommendations generated by SVD:")
        for idx, m_id in enumerate(recs_5):
            movie_title = movies_df[movies_df["MovieID"] == m_id]["Title"].values[0]
            movie_year = movies_df[movies_df["MovieID"] == m_id]["Year"].values[0]
            pred_score = svd_model.predict(sample_user, m_id)
            print(f"  {idx+1}. {movie_title} ({movie_year}) | Predicted Rating: {pred_score:.2f}")

if __name__ == "__main__":
    run_evaluation_pipeline(data_dir="data", max_test_users=100)
