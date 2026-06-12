# 🎬 Netflix Recommendation System

**Submitted By**

* **Yuvraj** (24124046)
* **Shreshtha Shrinivas** (24124040)

🌐 **Live Demo:** https://net-rec-sys.netlify.app/

---

## 📖 Project Overview

The **Netflix Recommendation System** is a full-stack movie recommendation platform inspired by the famous Netflix Prize Challenge. The system analyzes user rating patterns and generates personalized movie recommendations using machine learning techniques.

By leveraging collaborative filtering methods, the platform predicts user preferences and suggests relevant movies based on historical rating data.

---

## ✨ Features

* Personalized movie recommendations
* User viewing history analysis
* Movie catalog exploration
* Responsive and interactive user interface
* Machine Learning-based recommendation engine
* REST API integration
* Real-time recommendation retrieval

---
<img width="923" height="424" alt="Screenshot 2026-06-12 224128" src="https://github.com/user-attachments/assets/76939f64-b352-4d6c-9519-f9b523bddf7e" />

---
<img width="899" height="432" alt="Screenshot 2026-06-12 224152" src="https://github.com/user-attachments/assets/3ceccb78-22fd-4832-8951-a94859c28bec" />

---
<img width="872" height="424" alt="Screenshot 2026-06-12 224207" src="https://github.com/user-attachments/assets/e5eda167-5598-4558-b147-d9da72e99c60" />

---
## 🛠️ Tech Stack

### Frontend

* React.js
* Vite
* HTML5
* CSS3
* JavaScript

### Backend

* Python
* FastAPI

### Machine Learning & Data Analysis

* Pandas
* NumPy
* Scikit-learn
* Jupyter Notebook

### Dataset

* Netflix Prize Dataset
* Movie Titles Dataset

---

## 📂 Project Structure

```text
Recommendation-system-for-Netflix/
│
├── backend/
│   ├── API services
│   ├── Recommendation engine
│   └── Data processing modules
│
├── frontend/
│   ├── React components
│   ├── Pages
│   └── API integration
│
├── Netflix_Recommender.ipynb
│   ├── Data preprocessing
│   ├── Exploratory Data Analysis
│   └── Model training
│
├── movie_titles.csv
│
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

* Python 3.x
* Node.js
* npm

### Clone the Repository

```bash
git clone https://github.com/Conqueror63/Recommendation-system-for-Netflix.git

cd Recommendation-system-for-Netflix
```

### Backend Setup

```bash
cd backend

pip install -r requirements.txt

uvicorn main:app --reload
```

Backend runs at:

```text
http://localhost:8000
```

### Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

Frontend runs at:

```text
http://localhost:5173
```

---

## 📊 How It Works

### 1. Data Preprocessing

* Cleaned Netflix rating data
* Processed movie title mappings
* Removed inconsistencies and missing values

### 2. Exploratory Data Analysis

* Rating distribution analysis
* Popular movie identification
* User activity analysis

### 3. Model Training

The recommendation engine uses collaborative filtering techniques to identify patterns in user preferences.

The model:

* Learns from historical ratings
* Identifies similar users and movies
* Predicts ratings for unseen movies

### 4. Recommendation Generation

For a selected user:

1. User history is retrieved
2. Predicted ratings are calculated
3. Top-N movie recommendations are generated

---

## 📈 Future Enhancements

* Hybrid Recommendation System
* Deep Learning-based Recommendations
* User Authentication
* Watchlist Feature
* Movie Poster Integration
* Explainable Recommendations
* Real-time Recommendation Updates

---

## 📸 Screenshots

### Home Page

![Home Page](screenshots/home.png)

### Recommendations

![Recommendations](screenshots/recommendations.png)

### User History

![User History](screenshots/history.png)

---

## 🎯 Learning Outcomes

Through this project, we gained practical experience in:

* Recommendation Systems
* Machine Learning Pipelines
* Data Preprocessing
* REST API Development
* React Frontend Development
* Full-Stack Integration
* Model Deployment using Netlify and Render

---

## 🌐 Deployment

### Frontend (Netlify)

https://net-rec-sys.netlify.app/

### Backend (Render)

Deployed using FastAPI and Render Web Services.

---

## 📜 License

This project was developed for academic and educational purposes.
