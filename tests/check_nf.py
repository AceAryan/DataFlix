import pandas as pd

nf_movies  = pd.read_csv("data/raw/netflix/Netflix_Dataset_Movie.csv")
nf_ratings = pd.read_csv("data/raw/netflix/Netflix_Dataset_Rating.csv")

print("Movies sample:")
print(nf_movies.head(10).to_string())
print(f"\nTotal movies: {len(nf_movies)}")
print(f"Movie_ID range: {nf_ratings['Movie_ID'].min()} — {nf_ratings['Movie_ID'].max()}")
print(f"\nRatings sample:")
print(nf_ratings.head(5).to_string())