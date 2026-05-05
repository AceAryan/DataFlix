# import torch
# avail = torch.cuda.is_available()
# print("CUDA available:", avail)
# if avail:
#     print("Device count:", torch.cuda.device_count())
#     print("Device name:", torch.cuda.get_device_name(0))
# else:
#     print("No CUDA device found.")

# import pandas as pd
# train = pd.read_csv("data/processed/train.csv")
# print(train["rating"].describe())
# print("Sample values:", train["rating"].head(10).tolist())

# import torch, json
# from pathlib import Path

# # Check what the model predicts before any training
# # (random init + bias fix should predict ~3.5 immediately)
# with torch.no_grad():
#     dummy_out = torch.zeros(10)  # simulates zero-init MLP
#     print("Without bias fix:", torch.clamp(dummy_out, 0.5, 5.0))
#     dummy_out_fixed = dummy_out + 3.536
#     print("With bias fix:", torch.clamp(dummy_out_fixed, 0.5, 5.0))
#     # Should print tensor([3.5360, 3.5360, ...])

# import pandas as pd
# train = pd.read_csv("data/processed/train.csv")
# val   = pd.read_csv("data/processed/val.csv")
# print("Train rating mean:", train["rating"].mean())
# print("Val rating mean:  ", val["rating"].mean())
# print("Val rating range: ", val["rating"].min(), "-", val["rating"].max())

# import pandas as pd
# train = pd.read_csv("data/processed/train.csv")
# val   = pd.read_csv("data/processed/val.csv")

# train_users = set(train["user_idx"])
# train_items = set(train["movie_idx"])
# val_users   = set(val["user_idx"])
# val_items   = set(val["movie_idx"])

# print(f"Val users not in train:  {len(val_users - train_users):,} / {len(val_users):,} ({100*len(val_users-train_users)/len(val_users):.1f}%)")
# print(f"Val items not in train:  {len(val_items - train_items):,} / {len(val_items):,} ({100*len(val_items-train_items)/len(val_items):.1f}%)")

# # Also check — is the split temporal or random?
# if "timestamp" in train.columns:
#     print(f"\nTrain timestamp range: {train['timestamp'].min()} → {train['timestamp'].max()}")
#     print(f"Val timestamp range:   {val['timestamp'].min()} → {val['timestamp'].max()}")
#     overlap = train[(train['timestamp'] >= val['timestamp'].min())]
#     print(f"Train rows after val start: {len(overlap):,}")

# import pandas as pd
# train = pd.read_csv("data/processed/train.csv")
# print(train["rating"].describe())
# print(train["rating"].head())

# import pandas as pd
# train = pd.read_csv("data/processed/train.csv")
# val   = pd.read_csv("data/processed/val.csv")

# print("=== TRAIN ===")
# print(train["rating"].describe())
# print("\n=== VAL ===")  
# print(val["rating"].describe())
# print("\nVal columns:", val.columns.tolist())
# print("Val sample:\n", val.head(3))

import pandas as pd
import re

movies = pd.read_csv("data/processed/movies_metadata.csv")

# MovieLens encodes year in the title: "Toy Story (1995)"
movies["year"] = movies["title"].str.extract(r'\((\d{4})\)').astype(float)

print("Year range:", movies["year"].min(), "—", movies["year"].max())
print("\nMovies by decade:")
movies["decade"] = (movies["year"] // 10 * 10).astype("Int64")
print(movies["decade"].value_counts().sort_index().to_string())

print(f"\nPre-2000:  {(movies['year'] < 2000).sum():,} movies  ({100*(movies['year'] < 2000).mean():.1f}%)")
print(f"Post-2000: {(movies['year'] >= 2000).sum():,} movies  ({100*(movies['year'] >= 2000).mean():.1f}%)")