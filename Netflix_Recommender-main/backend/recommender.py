import os
import csv
import json
import numpy as np
def cosine_similarity(a, b):
    dot_product = np.dot(b, a.T).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    norms = norm_a * norm_b
    norms[norms == 0] = 1e-10
    return (dot_product / norms).reshape(1, -1)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

class RecommenderEngine:
    def __init__(self):
        self.P = None
        self.Q = None
        self.bu = None
        self.bi = None
        self.global_mean = 0
        self.user2idx = {}
        self.movie2idx = {}
        self.idx2movie = {}
        self.titles = {}
        self.stats = {}
        self.is_loaded = False

    def load_data(self):
        if self.is_loaded:
            return

        # Load SVD model factors
        npz_path = os.path.join(DATA_DIR, 'svd_model_factors.npz')
        if os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            self.P = data['P']
            self.Q = data['Q']
            self.bu = data['bu']
            self.bi = data['bi']
            self.global_mean = data['global_mean'][0]
            self.user2idx = dict(data['user2idx'])
            self.movie2idx = dict(data['movie2idx'])
            self.idx2movie = {v: k for k, v in self.movie2idx.items()}
            print(f"Loaded SVD Model: {len(self.user2idx)} users, {len(self.movie2idx)} movies.")
        else:
            print("WARNING: svd_model_factors.npz not found.")

        # Load Movie Titles
        csv_path = os.path.join(DATA_DIR, 'movie_titles.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3:
                        m_id = int(row[0])
                        year = row[1] if row[1] else "Unknown"
                        title = row[2]
                        self.titles[m_id] = {"title": title, "year": year}
            print(f"Loaded {len(self.titles)} movie titles.")
        else:
            print("WARNING: movie_titles.csv not found.")
            
        # Load Stats
        json_path = os.path.join(DATA_DIR, 'recommendation_results.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.stats = json.load(f)
                
        self.is_loaded = True

    def get_movie_details(self, movie_id):
        # Convert int/str
        movie_id = int(movie_id)
        if movie_id in self.titles:
            return self.titles[movie_id]
        return {"title": f"Unknown Movie {movie_id}", "year": ""}

    def get_explanation(self, movie_idx):
        """Generates an explanation by finding the most similar movies in the latent space."""
        if self.Q is None:
            return "Based on global popularity."
            
        target_q = self.Q[movie_idx].reshape(1, -1)
        sims = cosine_similarity(target_q, self.Q)[0]
        
        # Get top 2 similar items (excluding itself)
        top_indices = np.argsort(sims)[::-1][1:3]
        
        similar_titles = []
        for idx in top_indices:
            real_m_id = self.idx2movie.get(idx)
            if real_m_id:
                details = self.get_movie_details(real_m_id)
                similar_titles.append(f"'{details['title']}'")
                
        if similar_titles:
            return f"Because it shares features with {' and '.join(similar_titles)}."
        return "Based on your mathematical preference profile."

    def recommend_popularity(self, K=10, exclude_ids=None):
        """Cold start: recommend globally highest scored movies based on bias."""
        if exclude_ids is None:
            exclude_ids = set()
            
        if self.bi is None:
            return []
            
        # Global mean + movie bias gives a general popularity score
        scores = self.global_mean + self.bi
        top_indices = np.argsort(scores)[::-1]
        
        recs = []
        for idx in top_indices:
            m_id = self.idx2movie[idx]
            if m_id not in exclude_ids:
                details = self.get_movie_details(m_id)
                recs.append({
                    "movie_id": m_id,
                    "title": details["title"],
                    "year": details["year"],
                    "predicted_rating": round(float(scores[idx]), 2),
                    "explanation": "Trending movie globally."
                })
                if len(recs) >= K:
                    break
        return recs

    def recommend_svd(self, user_id, K=10, exclude_ids=None):
        """Generate recommendations using the pre-trained SVD model matrices."""
        if user_id not in self.user2idx or self.P is None:
            return self.recommend_popularity(K, exclude_ids)
            
        if exclude_ids is None:
            exclude_ids = set()
            
        u_idx = self.user2idx[user_id]
        
        # Fast vectorized scoring for all items
        # scores = global_mean + user_bias + item_bias + user_vector * item_matrix
        scores = (self.global_mean + 
                  self.bu[u_idx] + 
                  self.bi + 
                  self.Q @ self.P[u_idx])
                  
        scores = np.clip(scores, 1.0, 5.0)
        top_indices = np.argsort(scores)[::-1]
        
        recs = []
        for idx in top_indices:
            m_id = self.idx2movie[idx]
            if m_id not in exclude_ids:
                details = self.get_movie_details(m_id)
                explanation = self.get_explanation(idx)
                recs.append({
                    "movie_id": int(m_id),
                    "title": details["title"],
                    "year": details["year"],
                    "predicted_rating": round(float(scores[idx]), 2),
                    "explanation": explanation
                })
                if len(recs) >= K:
                    break
        return recs

    def recommend_item_cf(self, recent_ratings, K=10, exclude_ids=None):
        """Uses Item-Based CF based on recently rated items in SQLite."""
        if not recent_ratings or self.Q is None:
            return self.recommend_popularity(K, exclude_ids)
            
        if exclude_ids is None:
            exclude_ids = set()
            
        # Find latent representations of items the user liked
        liked_indices = []
        for r in recent_ratings:
            m_id = r['movie_id']
            if r['rating'] >= 3.5 and m_id in self.movie2idx:
                liked_indices.append(self.movie2idx[m_id])
                
        if not liked_indices:
            return self.recommend_popularity(K, exclude_ids)
            
        # Get mean vector of liked items
        user_proxy_vector = np.mean(self.Q[liked_indices], axis=0)
        
        # Compute cosine similarity between proxy vector and all items
        sims = cosine_similarity(user_proxy_vector.reshape(1, -1), self.Q)[0]
        
        # We also want to scale by global popularity to avoid recommending obscure weird items
        # just because they share a latent direction.
        item_scores = sims + (self.bi * 0.2) # small popularity boost
        
        top_indices = np.argsort(item_scores)[::-1]
        
        recs = []
        for idx in top_indices:
            m_id = self.idx2movie[idx]
            if m_id not in exclude_ids and idx not in liked_indices:
                details = self.get_movie_details(m_id)
                recs.append({
                    "movie_id": int(m_id),
                    "title": details["title"],
                    "year": details["year"],
                    "predicted_rating": round(float(np.clip(self.global_mean + self.bi[idx] + 0.5, 1, 5)), 2), # proxy rating
                    "explanation": "Based on items you recently rated."
                })
                if len(recs) >= K:
                    break
        return recs

# Singleton instance
engine = RecommenderEngine()
