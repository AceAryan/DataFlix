import numpy as np
import pandas as pd

als  = np.load("results/als_factors.npz")
train_df = pd.read_csv("data/processed/train.csv")

# Check user 0's mean
u = 0
user_id = train_df[train_df["user_idx"] == u]["user_id"].iloc[0]
user_mean = train_df[train_df["user_idx"] == u]["rating"].mean()

# Score
scores = (als["item_factors"] @ als["user_factors"][u]
          + als["item_biases"]
          + als["user_biases"][u]
          + user_mean)

print(f"User mean       : {user_mean:.3f}")
print(f"Score range     : {scores.min():.3f} to {scores.max():.3f}")
print(f"Score mean      : {scores.mean():.3f}")