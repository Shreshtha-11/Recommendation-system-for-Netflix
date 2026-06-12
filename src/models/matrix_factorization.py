import numpy as np
import pandas as pd
from .base import BaseRecommender

class FunkSVD(BaseRecommender):
    """
    Funk SVD (Stochastic Gradient Descent Matrix Factorization) recommender.
    Predicts rating as:
        r_ui = global_mean + user_bias[u] + movie_bias[i] + p_u . q_i
    """
    
    def __init__(self, n_factors=15, lr=0.005, reg=0.02, n_epochs=15, patience=3, init_mean=0, init_std=0.1):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.patience = patience
        self.init_mean = init_mean
        self.init_std = init_std
        
        self.global_mean = 3.5
        self.user_biases = None
        self.movie_biases = None
        self.p = None # User factors (N x d)
        self.q = None # Movie factors (M x d)
        
        self.user_id_map = {}
        self.movie_id_map = {}
        self.inv_user_map = {}
        self.inv_movie_map = {}
        self.epoch_loss_history = []
        
    def fit(self, train_df, val_df=None):
        self.epoch_loss_history = []
        self.global_mean = train_df["Rating"].mean()
        
        # Build mappings
        unique_users = train_df["UserID"].unique()
        unique_movies = train_df["MovieID"].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.inv_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.inv_movie_map = {idx: mid for mid, idx in self.movie_id_map.items()}
        
        n_users = len(unique_users)
        n_movies = len(unique_movies)
        
        # Initialize biases and vectors
        self.user_biases = np.zeros(n_users)
        self.movie_biases = np.zeros(n_movies)
        self.p = np.random.normal(self.init_mean, self.init_std / np.sqrt(self.n_factors), (n_users, self.n_factors))
        self.q = np.random.normal(self.init_mean, self.init_std / np.sqrt(self.n_factors), (n_movies, self.n_factors))
        
        u_indices = train_df["UserID"].map(self.user_id_map).values
        m_indices = train_df["MovieID"].map(self.movie_id_map).values
        ratings = train_df["Rating"].values.astype(float)
        
        print(f"Training FunkSVD (latent factors={self.n_factors}, epochs={self.n_epochs}, lr={self.lr}, reg={self.reg})...")
        
        best_val_rmse = float('inf')
        best_p = None
        best_q = None
        best_user_biases = None
        best_movie_biases = None
        no_improvement_epochs = 0
        
        for epoch in range(self.n_epochs):
            # Stochastic Gradient Descent shuffle
            shuffled_indices = np.random.permutation(len(ratings))
            epoch_loss = 0.0
            
            for idx in shuffled_indices:
                u = u_indices[idx]
                i = m_indices[idx]
                r = ratings[idx]
                
                # Rating prediction
                pred = self.global_mean + self.user_biases[u] + self.movie_biases[i] + np.dot(self.p[u], self.q[i])
                err = r - pred
                epoch_loss += err ** 2
                
                # Stochastic gradient updates
                self.user_biases[u] += self.lr * (err - self.reg * self.user_biases[u])
                self.movie_biases[i] += self.lr * (err - self.reg * self.movie_biases[i])
                
                p_u_temp = self.p[u].copy()
                self.p[u] += self.lr * (err * self.q[i] - self.reg * self.p[u])
                self.q[i] += self.lr * (err * p_u_temp - self.reg * self.q[i])
                
            train_rmse = np.sqrt(epoch_loss / len(ratings))
            self.epoch_loss_history.append(train_rmse)
            
            val_info = ""
            if val_df is not None:
                val_rmse = self._evaluate_rmse(val_df)
                val_info = f" | Val RMSE: {val_rmse:.4f}"
                
                # Early stopping check
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_p = self.p.copy()
                    best_q = self.q.copy()
                    best_user_biases = self.user_biases.copy()
                    best_movie_biases = self.movie_biases.copy()
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1
                    
            print(f"Epoch {epoch+1}/{self.n_epochs} - Train RMSE: {train_rmse:.4f}{val_info}")
            
            if val_df is not None and no_improvement_epochs >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Restoring best weights (Val RMSE: {best_val_rmse:.4f}).")
                self.p = best_p
                self.q = best_q
                self.user_biases = best_user_biases
                self.movie_biases = best_movie_biases
                break

    def predict(self, user_id, movie_id):
        user_known = user_id in self.user_id_map
        movie_known = movie_id in self.movie_id_map
        
        if not user_known and not movie_known:
            return self.global_mean
        elif not user_known:
            m_idx = self.movie_id_map[movie_id]
            return self.global_mean + self.movie_biases[m_idx]
        elif not movie_known:
            u_idx = self.user_id_map[user_id]
            return self.global_mean + self.user_biases[u_idx]
            
        u_idx = self.user_id_map[user_id]
        m_idx = self.movie_id_map[movie_id]
        
        pred = self.global_mean + self.user_biases[u_idx] + self.movie_biases[m_idx] + np.dot(self.p[u_idx], self.q[m_idx])
        return float(np.clip(pred, 1.0, 5.0))
        
    def _evaluate_rmse(self, df):
        preds = []
        for _, row in df.iterrows():
            preds.append(self.predict(row["UserID"], row["MovieID"]))
        return np.sqrt(np.mean((df["Rating"].values - np.array(preds)) ** 2))
        
    def get_similar_items(self, movie_id, n=10):
        """
        Calculates similarity via cosine distance of item latent factor vectors.
        """
        if movie_id not in self.movie_id_map:
            return []
            
        m_idx = self.movie_id_map[movie_id]
        q_target = self.q[m_idx]
        q_target_norm = np.linalg.norm(q_target)
        
        if q_target_norm == 0:
            return []
            
        similarities = []
        for i_idx, movie_factor in enumerate(self.q):
            if i_idx == m_idx:
                continue
            m_norm = np.linalg.norm(movie_factor)
            if m_norm == 0:
                continue
            sim = np.dot(q_target, movie_factor) / (q_target_norm * m_norm)
            similarities.append((self.inv_movie_map[i_idx], float(sim)))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
