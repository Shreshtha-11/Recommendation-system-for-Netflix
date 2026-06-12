import os
import pandas as pd
import numpy as np

def load_movie_titles(file_path):
    """
    Parses the Netflix movie titles text file.
    Format is MovieID,Year,Title.
    Handles titles that contain commas by splitting only on the first two commas.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Movie titles file not found at: {file_path}")
        
    records = []
    encoding = "latin-1" # Prevent encoding errors on Windows
    
    with open(file_path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) >= 3:
                movie_id = int(parts[0])
                year_str = parts[1].strip()
                year = int(year_str) if year_str.isdigit() else None
                title = parts[2].strip()
                records.append((movie_id, year, title))
            elif len(parts) == 2:
                movie_id = int(parts[0])
                title = parts[1].strip()
                records.append((movie_id, None, title))
                
    return pd.DataFrame(records, columns=["MovieID", "Year", "Title"])

def load_ratings_from_file(file_path, max_ratings=None):
    """
    Parses a single Netflix combined_data_*.txt file.
    Format:
    MovieID:
    UserID,Rating,Date
    UserID,Rating,Date
    
    Parameters:
    -----------
    file_path : str
        Path to the text file.
    max_ratings : int, optional
        Maximum number of ratings to load (useful for CPU/time constraints).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ratings file not found at: {file_path}")
        
    user_ids = []
    movie_ids = []
    ratings = []
    dates = []
    
    current_movie_id = None
    ratings_count = 0
    
    print(f"Parsing ratings from '{file_path}'...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.endswith(":"):
                current_movie_id = int(line[:-1])
            else:
                parts = line.split(",")
                if len(parts) == 3:
                    user_ids.append(int(parts[0]))
                    movie_ids.append(current_movie_id)
                    ratings.append(int(parts[1]))
                    dates.append(parts[2])
                    
                    ratings_count += 1
                    if max_ratings and ratings_count >= max_ratings:
                        break
                        
    df = pd.DataFrame({
        "UserID": user_ids,
        "MovieID": movie_ids,
        "Rating": ratings,
        "Date": pd.to_datetime(dates)
    })
    
    print(f"Successfully loaded {len(df)} ratings.")
    return df

def split_data_chronologically(df, test_ratio=0.2):
    """
    Performs a user-stratified chronological split:
    For each user, ratings are sorted by Date, and the most recent
    'test_ratio' ratings are placed in the test set.
    """
    print(f"Splitting dataset chronologically (test_ratio={test_ratio})...")
    
    # Sort by UserID and Date
    df_sorted = df.sort_values(by=["UserID", "Date"])
    
    # Calculate counts and cumulative counts
    user_counts = df_sorted.groupby("UserID").size()
    cum_count = df_sorted.groupby("UserID").cumcount()
    user_total = df_sorted["UserID"].map(user_counts)
    
    # Last test_ratio portion goes to test
    test_indices = cum_count >= (user_total * (1 - test_ratio))
    
    # Ensure every user has at least 1 rating in training (cum_count > 0)
    test_indices = test_indices & (cum_count > 0)
    
    train_df = df_sorted[~test_indices].copy()
    test_df = df_sorted[test_indices].copy()
    
    print(f"Split complete. Train size: {len(train_df)} | Test size: {len(test_df)}")
    return train_df, test_df

def get_user_interactions(train_df):
    """
    Returns a dictionary mapping UserID -> set of MovieIDs they rated in training.
    """
    return train_df.groupby("UserID")["MovieID"].apply(set).to_dict()
