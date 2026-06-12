import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from src.data_utils import load_movie_titles, get_user_interactions
from src.models.collaborative_filtering import ItemCollaborativeFiltering
from src.models.matrix_factorization import FunkSVD
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Cinemix Recommendation API",
    description="Backend API powering the Netflix Recommendation System dashboard",
    version="1.0.0"
)

# Enable CORS for React frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory caches for data and models
DATA_DIR = "data"
movies_df = None
ratings_df = None
train_interactions = None
svd_model = None
item_cf_model = None
all_movie_ids = []

@app.on_event("startup")
async def startup_event():
    global movies_df, ratings_df, train_interactions, svd_model, item_cf_model, all_movie_ids
    
    print("Loading datasets and fitting recommendation models...")
    
    movie_titles_path = os.path.join(DATA_DIR, "movie_titles.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")
    
    if not os.path.exists(movie_titles_path) or not os.path.exists(ratings_path):
        print("Required files not found. Startup halted.")
        return
        
    movies_df = load_movie_titles(movie_titles_path)
    ratings_df = pd.read_csv(ratings_path)
    ratings_df["Date"] = pd.to_datetime(ratings_df["Date"])
    
    # User-stratified split or train on full subset
    # Since ratings.csv is already a capped development subset of 100k, we train on the full ratings_df
    train_interactions = get_user_interactions(ratings_df)
    all_movie_ids = list(ratings_df["MovieID"].unique())
    
    # Fit FunkSVD
    svd_model = FunkSVD(n_factors=15, lr=0.005, reg=0.02, n_epochs=10)
    svd_model.fit(ratings_df)
    
    # Fit Item CF for similarity neighborhood queries
    item_cf_model = ItemCollaborativeFiltering(k=20)
    item_cf_model.fit(ratings_df)
    
    print("Models fitted successfully. API ready.")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "loaded_movies": len(movies_df) if movies_df is not None else 0}

@app.get("/api/stats")
async def get_system_stats():
    if ratings_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
        
    n_ratings = len(ratings_df)
    n_users = int(ratings_df["UserID"].nunique())
    n_movies = int(ratings_df["MovieID"].nunique())
    
    # Sparsity
    sparsity = 1.0 - (n_ratings / (n_users * n_movies))
    
    # Rating counts
    dist = ratings_df["Rating"].value_counts().sort_index().to_dict()
    # Ensure all stars 1-5 are present
    rating_distribution = {str(star): int(dist.get(star, 0)) for star in range(1, 6)}
    
    # Average rating
    avg_rating = float(ratings_df["Rating"].mean())
    
    return {
        "total_ratings": n_ratings,
        "unique_users": n_users,
        "unique_movies": n_movies,
        "sparsity": float(sparsity),
        "average_rating": avg_rating,
        "rating_distribution": rating_distribution
    }

@app.get("/api/movies")
async def get_movies_list():
    if movies_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    # Return list of movies present in dataset
    unique_dataset_movies = set(ratings_df["MovieID"].unique())
    active_movies = movies_df[movies_df["MovieID"].isin(unique_dataset_movies)]
    
    records = []
    for _, row in active_movies.iterrows():
        records.append({
            "movie_id": int(row["MovieID"]),
            "year": int(row["Year"]) if pd.notna(row["Year"]) else None,
            "title": str(row["Title"])
        })
    return records

@app.get("/api/user/{user_id}/history")
async def get_user_history(user_id: int):
    if ratings_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
        
    user_ratings = ratings_df[ratings_df["UserID"] == user_id]
    if user_ratings.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} has no interaction history")
        
    user_history = user_ratings.merge(movies_df, on="MovieID").sort_values(by="Rating", ascending=False)
    
    history_list = []
    for _, row in user_history.iterrows():
        history_list.append({
            "movie_id": int(row["MovieID"]),
            "title": str(row["Title"]),
            "year": int(row["Year"]) if pd.notna(row["Year"]) else None,
            "rating": int(row["Rating"]),
            "date": row["Date"].strftime("%Y-%m-%d")
        })
    return history_list

@app.get("/api/user/{user_id}/recommend")
async def get_user_recommendations(user_id: int, n: int = 10):
    if svd_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    # Check if user is known (present in data)
    user_known = user_id in svd_model.user_id_map
    watched_items = train_interactions.get(user_id, set()) if user_known else set()
    
    # Generate recommendations using SVD
    recommended_ids = svd_model.recommend(user_id, n=n, train_interactions=watched_items, all_movie_ids=all_movie_ids)
    
    recs = []
    for idx, m_id in enumerate(recommended_ids):
        m_info = movies_df[movies_df["MovieID"] == m_id]
        if m_info.empty:
            continue
        title = str(m_info["Title"].values[0])
        year = int(m_info["Year"].values[0]) if pd.notna(m_info["Year"].values[0]) else None
        pred_rating = svd_model.predict(user_id, m_id)
        
        # Simple similarity explanation hook (e.g. similar to user's favorite movie in history)
        explanation = "Recommended based on your latent profile"
        if user_known and len(watched_items) > 0:
            sims = svd_model.get_similar_items(m_id, n=5)
            for sim_id, _ in sims:
                if sim_id in watched_items:
                    fav_title = movies_df[movies_df["MovieID"] == sim_id]["Title"].values[0]
                    explanation = f"Similar to your favorite movie: '{fav_title}'"
                    break
                    
        recs.append({
            "rank": idx + 1,
            "movie_id": int(m_id),
            "title": title,
            "year": year,
            "predicted_rating": float(round(pred_rating, 2)),
            "explanation": explanation
        })
    return recs

@app.get("/api/movie/{movie_id}/similar")
async def get_similar_movies(movie_id: int, n: int = 10):
    if svd_model is None or item_cf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    m_info = movies_df[movies_df["MovieID"] == movie_id]
    if m_info.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
        
    svd_sim = svd_model.get_similar_items(movie_id, n=n)
    cf_sim = item_cf_model.get_similar_items(movie_id, n=n)
    
    svd_list = []
    for m_id, score in svd_sim:
        sim_m = movies_df[movies_df["MovieID"] == m_id]
        if not sim_m.empty:
            svd_list.append({
                "movie_id": int(m_id),
                "title": str(sim_m["Title"].values[0]),
                "year": int(sim_m["Year"].values[0]) if pd.notna(sim_m["Year"].values[0]) else None,
                "score": float(score)
            })
            
    cf_list = []
    for m_id, score in cf_sim:
        sim_m = movies_df[movies_df["MovieID"] == m_id]
        if not sim_m.empty:
            cf_list.append({
                "movie_id": int(m_id),
                "title": str(sim_m["Title"].values[0]),
                "year": int(sim_m["Year"].values[0]) if pd.notna(sim_m["Year"].values[0]) else None,
                "score": float(score)
            })
            
    return {
        "movie_title": str(m_info["Title"].values[0]),
        "svd_similar": svd_list,
        "cf_similar": cf_list
    }
