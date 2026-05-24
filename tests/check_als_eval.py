import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("D:/Dev/DataFlix")
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_CSV, TEST_CSV, ALS_PATH
from src.models.als import ALS

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
als      = ALS.load(ALS_PATH)

# Pick user 0
user_idx = int(train_df["user_idx"].iloc[0])

# What they rated in test
test_user = test_df[test_df["user_idx"] == user_idx]
print(f"User {user_idx} — {len(test_user)} test ratings")
print(f"  rating_centered range: {test_user['rating_centered'].min():.2f} to {test_user['rating_centered'].max():.2f}")

# How many are "relevant" (above mean)
relevant = set(test_user[test_user["rating_centered"] > 0]["movie_idx"].astype(int).tolist())
print(f"  Relevant items (rc > 0): {len(relevant)}")

# Score all items
scores = als.item_factors @ als.user_factors[user_idx] + als.item_biases + als.user_biases[user_idx]

# What are the scores of relevant items?
if relevant:
    rel_scores = [scores[m] for m in relevant]
    print(f"  Scores of relevant items: min={min(rel_scores):.3f} max={max(rel_scores):.3f}")

# What rank do relevant items appear at?
seen = set(train_df[train_df["user_idx"] == user_idx]["movie_idx"].astype(int).tolist())
scores[list(seen)] = -np.inf
ranked = np.argsort(scores)[::-1]
for rel_item in list(relevant)[:3]:
    rank = int(np.where(ranked == rel_item)[0][0]) + 1
    print(f"  Item {rel_item} rank: {rank} / {len(scores)}")