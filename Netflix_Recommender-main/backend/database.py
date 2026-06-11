import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ratings.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            user_id INTEGER,
            movie_id INTEGER,
            rating REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, movie_id)
        )
    ''')
    conn.commit()
    conn.close()

def add_rating(user_id: int, movie_id: int, rating: float):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (user_id, movie_id, rating, datetime.now()))
    conn.commit()
    conn.close()

def get_user_ratings(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT movie_id, rating FROM ratings WHERE user_id = ? ORDER BY timestamp DESC
    ''', (user_id,))
    results = c.fetchall()
    conn.close()
    return [{"movie_id": row[0], "rating": row[1]} for row in results]

# Initialize on import
init_db()
