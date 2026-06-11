from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from recommender import engine
from database import add_rating, get_user_ratings

app = Flask(__name__)
# Allow CORS for frontend
CORS(app)

# Load data on startup
engine.load_data()

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    user_id = data.get("user_id")
    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400
        
    try:
        user_id = int(user_id)
    except ValueError:
        return jsonify({"error": "user_id must be an integer"}), 400

    # Check if user exists in the core dataset
    is_existing = user_id in engine.user2idx
    
    # Also check if they have ratings in SQLite
    recent = get_user_ratings(user_id)
    has_recent = len(recent) > 0
    
    msg = "Welcome back!" if is_existing else ("Welcome! New user profile created." if not has_recent else "Welcome back!")
    
    return jsonify({
        "user_id": user_id,
        "is_existing_svd_user": is_existing,
        "has_recent_ratings": has_recent,
        "message": msg
    })

@app.route("/api/recommendations/<int:user_id>", methods=["GET"])
def get_recommendations(user_id):
    k = request.args.get("k", 10, type=int)
    
    # Get any recent ratings from SQLite to exclude them from recs
    recent_ratings = get_user_ratings(user_id)
    exclude_ids = set([r['movie_id'] for r in recent_ratings])
    
    # 1. Existing SVD User
    if user_id in engine.user2idx:
        recs = engine.recommend_svd(user_id, K=k, exclude_ids=exclude_ids)
    # 2. New User, but has submitted ratings online
    elif len(recent_ratings) > 0:
        recs = engine.recommend_item_cf(recent_ratings, K=k, exclude_ids=exclude_ids)
    # 3. Brand New User (Cold Start)
    else:
        recs = engine.recommend_popularity(K=k, exclude_ids=exclude_ids)
        
    return jsonify({
        "user_id": user_id,
        "recommendations": recs
    })

@app.route("/api/rate", methods=["POST"])
def rate_movie():
    data = request.json
    user_id = data.get("user_id")
    movie_id = data.get("movie_id")
    rating = data.get("rating")
    
    if None in (user_id, movie_id, rating):
        return jsonify({"error": "Missing fields"}), 400
        
    try:
        user_id = int(user_id)
        movie_id = int(movie_id)
        rating = float(rating)
    except ValueError:
        return jsonify({"error": "Invalid data types"}), 400
        
    if rating < 1 or rating > 5:
        return jsonify({"error": "Rating must be between 1 and 5"}), 400
        
    add_rating(user_id, movie_id, rating)
    return jsonify({"success": True, "message": "Rating saved"})

@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify(engine.stats)

@app.route("/api/movies/popular", methods=["GET"])
def get_popular_movies():
    k = request.args.get("k", 50, type=int)
    return jsonify(engine.recommend_popularity(K=k))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
