import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR

print("Testing Path B data loading...")
try:
    with open(PROCESSED_DIR / "user_positives.pkl", "rb") as f:
        user_positives = pickle.load(f)
    print(f"Loaded user_positives: {len(user_positives)} users")
    
    bpr_data = np.load(PROCESSED_DIR / "bpr_data.npz")
    all_items = bpr_data["all_items"]
    print(f"Loaded all_items: {len(all_items)} items")
    
    # Check for genre table and other features
    genre_table = torch.load(PROCESSED_DIR / "genre_table.pt", weights_only=False)
    print("Genre table loaded.")
    
    print("Path B data loading SUCCESSFUL.")
except Exception as e:
    print(f"Path B data loading FAILED: {e}")
