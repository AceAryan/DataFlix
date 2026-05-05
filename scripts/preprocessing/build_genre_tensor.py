import torch
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

# Load
data = torch.load(PROCESSED_DIR / "genre_table.pt")

movie_genre_ids = data["movie_genre_ids"]   # <-- IMPORTANT
n_genres = data["n_genres"]

# Infer number of movies
# movie_genre_ids is likely dict: {movie_id: [genre_ids]}
n_movies = max(movie_genre_ids.keys()) + 1

# Create multi-hot tensor
genre_tensor = torch.zeros(n_movies, n_genres, dtype=torch.float32)

for movie_id, genres in movie_genre_ids.items():
    if genres is None or len(genres) == 0:
        continue
    genre_tensor[movie_id, genres] = 1.0

# Save
save_path = PROCESSED_DIR / "genre_tensor.pt"
torch.save(genre_tensor, save_path)

print("Saved:", save_path)
print("Shape:", genre_tensor.shape)