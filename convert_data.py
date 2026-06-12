import pandas as pd

def parse_sampled_file(file_path, max_movies=500, ratings_per_movie=100):
    data = []
    current_movie = None
    movie_count = 0
    ratings_count = 0
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                current_movie = int(line.replace(":", ""))
                movie_count += 1
                ratings_count = 0
                if movie_count > max_movies:
                    break
            else:
                if ratings_count < ratings_per_movie:
                    parts = line.split(",")
                    if len(parts) == 3:
                        user_id, rating, date = parts
                        data.append([
                            int(user_id),
                            current_movie,
                            int(rating),
                            date
                        ])
                        ratings_count += 1
    return data

print("Sampling movies from combined data 1 and 2...")
data1 = parse_sampled_file("data/combined_data_1.txt", 500, 100)
data2 = parse_sampled_file("data/combined_data_2.txt", 500, 100)

combined_data = data1 + data2

df = pd.DataFrame(
    combined_data,
    columns=["UserID", "MovieID", "Rating", "Date"]
)

df.to_csv("data/ratings.csv", index=False)

print(f"Extraction completed. Extracted {df['MovieID'].nunique()} movies and {len(df)} ratings.")
print(df.head())