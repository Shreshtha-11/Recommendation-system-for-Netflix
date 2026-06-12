# Submitted by - Yuvraj 24124046 , Shreshtha Shrinivas 24124040

##Project Deployed over Netlify: https://net-rec-sys.netlify.app/

#Netflix Recommendation System
This project is a full-stack implementation of a movie recommendation engine, inspired by the classic Netflix Prize challenge. It uses machine learning to predict user movie preferences and provide personalized recommendations.

##🚀 Project Overview
This system processes movie rating data to suggest titles to users. It leverages [mention specific techniques, e.g., Collaborative Filtering/Matrix Factorization] to analyze user viewing patterns and predict ratings for unwatched content.

<img width="923" height="424" alt="Screenshot 2026-06-12 224128" src="https://github.com/user-attachments/assets/76939f64-b352-4d6c-9519-f9b523bddf7e" />

<img width="899" height="432" alt="Screenshot 2026-06-12 224152" src="https://github.com/user-attachments/assets/3ceccb78-22fd-4832-8951-a94859c28bec" />

<img width="872" height="424" alt="Screenshot 2026-06-12 224207" src="https://github.com/user-attachments/assets/e5eda167-5598-4558-b147-d9da72e99c60" />


##🛠️ Tech Stack
Frontend: React, Vite

Backend: [e.g., Python, Flask/FastAPI]

Machine Learning: Scikit-learn, Pandas, NumPy, Jupyter Notebooks

Dataset: Netflix Prize Dataset (Movie Ratings)

##📁 Repository Structure
/backend: API and logic for serving recommendations.

/frontend: The user interface for interacting with the recommender.

/Netflix_Recommender.ipynb: The primary notebook where data analysis and model training occurred.

movie_titles.csv: Mapping of movie IDs to their respective titles.

##⚙️ Installation & Setup
Prerequisites
Node.js (for the frontend)

Python 3.x (for the backend/model)

###Steps
Clone the repository:

####Bash
git clone https://github.com/Conqueror63/Recommendation-system-for-Netflix.git
cd Recommendation-system-for-Netflix
Run the Backend:

####Bash
cd backend
# Install dependencies and start your server here
Run the Frontend:

####Bash
cd ../frontend
npm install
npm run dev


##📊 How It Works
###Data Preprocessing: Cleaned the raw Netflix rating data and handled movie title mappings.

###Model Training: Used [mention your model, e.g., SVD or KNN] to identify patterns in user ratings.

###Prediction: The system calculates the similarity between users/items and generates top-N recommendations.
