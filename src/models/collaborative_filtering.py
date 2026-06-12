import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from .base import BaseRecommender

class UserCollaborativeFiltering(BaseRecommender):
    """
    User-Based Collaborative Filtering using Cosine Similarity.
    Includes user bias correction.
    """
    
    def __init__(self, k=20, min_support=1):
        self.k = k
        self.min_support = min_support
        self.global_mean = 3.5
        self.user_means = {}
        self.user_id_map = {}
        self.movie_id_map = {}
        self.ratings_matrix = None
        self.ratings_matrix_csc = None
        self.user_norms = None
        
    def fit(self, train_df):
        self.global_mean = train_df["Rating"].mean()
        self.user_means = train_df.groupby("UserID")["Rating"].mean().to_dict()
        
        unique_users = train_df["UserID"].unique()
        unique_movies = train_df["MovieID"].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.inv_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.inv_movie_map = {idx: mid for mid, idx in self.movie_id_map.items()}
        
        # Build SciPy sparse matrix
        rows = train_df["UserID"].map(self.user_id_map).values
        cols = train_df["MovieID"].map(self.movie_id_map).values
        
        # Mean center ratings
        user_average_ratings = train_df["UserID"].map(self.user_means).values
        centered_ratings = train_df["Rating"].values - user_average_ratings
        
        self.ratings_matrix = csr_matrix(
            (centered_ratings, (rows, cols)),
            shape=(len(unique_users), len(unique_movies))
        )
        
        # Precompute norms for cosine similarity
        norms = np.sqrt(np.asarray(self.ratings_matrix.power(2).sum(axis=1)).reshape(-1))
        norms[norms == 0] = 1.0 # Avoid division by zero
        self.user_norms = norms
        self.ratings_matrix_csc = self.ratings_matrix.tocsc()

    def predict(self, user_id, movie_id):
        if user_id not in self.user_id_map:
            return self.global_mean
        if movie_id not in self.movie_id_map:
            return self.user_means.get(user_id, self.global_mean)
            
        u_idx = self.user_id_map[user_id]
        m_idx = self.movie_id_map[movie_id]
        
        # Find all users who rated this movie
        matching_rows = self.ratings_matrix_csc[:, m_idx]
        neighbor_indices = matching_rows.nonzero()[0]
        
        # Exclude self
        neighbor_indices = neighbor_indices[neighbor_indices != u_idx]
        
        if len(neighbor_indices) == 0:
            return self.user_means[user_id]
            
        # Diffs for co-rated movie
        neighbor_ratings = matching_rows[neighbor_indices, 0].toarray().reshape(-1)
        
        # Compute cosine similarity for all neighbors simultaneously
        neighbor_matrix = self.ratings_matrix[neighbor_indices]
        u_row = self.ratings_matrix[u_idx]
        dot_products = neighbor_matrix.dot(u_row.T).toarray().reshape(-1)
        
        n_norms = self.user_norms[neighbor_indices]
        u_norm = self.user_norms[u_idx]
        denom = u_norm * n_norms
        denom[denom == 0] = 1.0
        
        sims = dot_products / denom
        
        # Filter for positive similarity neighbors
        pos_mask = sims > 0
        if not np.any(pos_mask):
            return self.user_means[user_id]
            
        sims = sims[pos_mask]
        neighbor_ratings = neighbor_ratings[pos_mask]
        
        # Sort similarities and pick top K
        if len(sims) > self.k:
            top_k = np.argsort(sims)[::-1][:self.k]
            sims = sims[top_k]
            neighbor_ratings = neighbor_ratings[top_k]
            
        sim_sum = np.sum(sims)
        weighted_sum = np.sum(sims * neighbor_ratings)
        
        if sim_sum == 0:
            return self.user_means[user_id]
            
        pred = self.user_means[user_id] + (weighted_sum / sim_sum)
        return float(np.clip(pred, 1.0, 5.0))


class ItemCollaborativeFiltering(BaseRecommender):
    """
    Item-Based Collaborative Filtering using Adjusted Cosine Similarity.
    """
    
    def __init__(self, k=20, min_support=1):
        self.k = k
        self.min_support = min_support
        self.global_mean = 3.5
        self.user_means = {}
        self.user_id_map = {}
        self.movie_id_map = {}
        self.similarity_matrix = None
        self.ratings_matrix = None
        self.ratings_matrix_csc = None
        
    def fit(self, train_df):
        self.global_mean = train_df["Rating"].mean()
        self.user_means = train_df.groupby("UserID")["Rating"].mean().to_dict()
        
        unique_users = train_df["UserID"].unique()
        unique_movies = train_df["MovieID"].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.inv_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.inv_movie_map = {idx: mid for mid, idx in self.movie_id_map.items()}
        
        # Ratings matrix with Movies as rows, Users as columns
        rows = train_df["MovieID"].map(self.movie_id_map).values
        cols = train_df["UserID"].map(self.user_id_map).values
        ratings = train_df["Rating"].values
        
        self.ratings_matrix = csr_matrix(
            (ratings, (rows, cols)),
            shape=(len(unique_movies), len(unique_users))
        )
        
        # Precompute Pearson centered matrix (Movies as rows)
        movie_means = np.asarray(self.ratings_matrix.sum(axis=1)).reshape(-1) / np.asarray((self.ratings_matrix > 0).sum(axis=1)).reshape(-1).clip(1)
        
        nz_rows, nz_cols = self.ratings_matrix.nonzero()
        centered_data = self.ratings_matrix.data - movie_means[nz_rows]
        
        centered_matrix = csr_matrix(
            (centered_data, (nz_rows, nz_cols)),
            shape=self.ratings_matrix.shape
        )
        
        # Calculate movie norms
        norms = np.sqrt(np.asarray(centered_matrix.power(2).sum(axis=1)).reshape(-1))
        norms[norms == 0] = 1.0
        
        # Compute Cosine Similarity between all movies
        dot_product = centered_matrix.dot(centered_matrix.T).toarray()
        norm_outer = np.outer(norms, norms)
        norm_outer[norm_outer == 0] = 1.0
        
        self.similarity_matrix = dot_product / norm_outer
        np.fill_diagonal(self.similarity_matrix, 0.0) # Mask self-similarity
        self.ratings_matrix_csc = self.ratings_matrix.tocsc()

    def predict(self, user_id, movie_id):
        if user_id not in self.user_id_map:
            return self.global_mean
        if movie_id not in self.movie_id_map:
            return self.user_means.get(user_id, self.global_mean)
            
        u_idx = self.user_id_map[user_id]
        m_idx = self.movie_id_map[movie_id]
        
        # Find all movies rated by this user
        user_ratings_col = self.ratings_matrix_csc[:, u_idx]
        rated_movie_indices = user_ratings_col.nonzero()[0]
        
        if len(rated_movie_indices) == 0:
            return self.user_means[user_id]
            
        # Get similarities and user ratings for co-rated items
        sims = self.similarity_matrix[m_idx, rated_movie_indices]
        user_ratings = user_ratings_col[rated_movie_indices, 0].toarray().reshape(-1)
        
        # Filter positive similarities
        pos_mask = sims > 0
        if not np.any(pos_mask):
            return self.user_means[user_id]
            
        sims = sims[pos_mask]
        user_ratings = user_ratings[pos_mask]
        
        # Take top K similar items
        if len(sims) > self.k:
            top_k = np.argsort(sims)[::-1][:self.k]
            sims = sims[top_k]
            user_ratings = user_ratings[top_k]
            
        sim_sum = np.sum(sims)
        weighted_sum = np.sum(sims * user_ratings)
        
        if sim_sum == 0:
            return self.user_means[user_id]
            
        pred = weighted_sum / sim_sum
        return float(np.clip(pred, 1.0, 5.0))
        
    def get_similar_items(self, movie_id, n=10):
        if movie_id not in self.movie_id_map:
            return []
        m_idx = self.movie_id_map[movie_id]
        sims = self.similarity_matrix[m_idx]
        
        similar_indices = np.argsort(sims)[::-1][:n]
        results = []
        for idx in similar_indices:
            score = sims[idx]
            if score <= 0:
                continue
            orig_id = self.inv_movie_map[idx]
            results.append((orig_id, float(score)))
        return results
