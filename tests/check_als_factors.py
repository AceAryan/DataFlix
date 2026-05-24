import numpy as np

als = np.load("results/als_factors.npz")
user_factors = als["user_factors"]
item_factors = als["item_factors"]
user_biases  = als["user_biases"]
item_biases  = als["item_biases"]

# Pick user 0 and score all items
u = 0
scores = item_factors @ user_factors[u] + item_biases + user_biases[u]

print("Score stats:")
print(f"  min  : {scores.min():.4f}")
print(f"  max  : {scores.max():.4f}")
print(f"  mean : {scores.mean():.4f}")
print(f"  std  : {scores.std():.4f}")
print(f"  top10 scores: {sorted(scores)[-10:]}")

# Check how many unique scores there are
unique = len(np.unique(scores.round(4)))
print(f"  unique scores (4dp): {unique} out of {len(scores)}")