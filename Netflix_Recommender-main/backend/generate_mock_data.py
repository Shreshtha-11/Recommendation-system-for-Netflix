import os
import json
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Mock SVD Factors
# Let's say we have 100 users and 50 movies, with k=10 latent factors
n_users = 100
n_movies = 50
n_factors = 10

np.random.seed(42)
P = np.random.normal(0, 0.1, (n_users, n_factors))
Q = np.random.normal(0, 0.1, (n_movies, n_factors))
bu = np.random.normal(0, 0.1, n_users)
bi = np.random.normal(0, 0.5, n_movies)
global_mean = np.array([3.5])

# Mappings: user IDs and movie IDs are just strings for flexibility, or ints
# In the original Netflix data, they are ints. Let's use 1..100 for users, 1..50 for movies.
user2idx = np.array([(i, i-1) for i in range(1, n_users + 1)], dtype=object)
movie2idx = np.array([(i, i-1) for i in range(1, n_movies + 1)], dtype=object)

np.savez_compressed(os.path.join(DATA_DIR, 'svd_model_factors.npz'),
                    P=P, Q=Q, bu=bu, bi=bi, 
                    global_mean=global_mean,
                    user2idx=user2idx,
                    movie2idx=movie2idx)
print("Mock svd_model_factors.npz generated.")

# 2. Mock Movie Titles
import csv
movie_titles_path = os.path.join(DATA_DIR, 'movie_titles.csv')
with open(movie_titles_path, 'w', newline='', encoding='utf-8') as f:
    # Netflix format: ID,Year,Title (no header usually, but we can just write data)
    writer = csv.writer(f)
    for i in range(1, n_movies + 1):
        writer.writerow([i, 2000 + (i % 25), f"Mock Movie Title {i}"])
print("Mock movie_titles.csv generated.")

# 3. Mock Recommendation Results
results = {
    "dataset_stats": {
        "n_users": 480000,
        "n_movies": 17000,
        "total_ratings": 100000000
    },
    "model_comparison": [
        {"Model": "SVD", "RMSE": 0.89, "MAP@10": 0.15},
        {"Model": "Baseline", "RMSE": 0.95, "MAP@10": 0.05}
    ]
}
with open(os.path.join(DATA_DIR, 'recommendation_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("Mock recommendation_results.json generated.")
