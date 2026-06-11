Cultural Council IITR Summer Open Project 2026
AIML PS
Netflix Movie Recommender System

The system starts with precomputed recommendation data stored in the backend. The main model is an SVD-based collaborative filtering model. For each user and movie, it stores latent vectors and bias values, then predicts how much a user will like each movie.

It also supports three recommendation modes:

Existing user in the SVD model
The app uses the trained SVD factors to generate personalized recommendations.
New user with some ratings
The app uses item-based collaborative filtering from the movies the user recently rated in the SQLite database.
Brand new user with no history
The app falls back to popular movies using global movie bias scores.
