import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import DEVICE, PROCESSED_DIR, RESULTS_DIR
from src.models.hybrid import DataFlixModel
from src.training.trainer import BPRDataset, collate_bpr
from torch.utils.data import DataLoader
import numpy as np
import pickle

def verify_path_b():
    print(f"Device: {DEVICE}")
    n_users = 305999
    n_items = 24383
    
    # Load model
    model = DataFlixModel(n_users, n_items, path="B").to(DEVICE)
    model_path = RESULTS_DIR / "dataflix_path_b.pt"
    
    if not model_path.exists():
        print(f"Model file {model_path} not found!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    print("Model loaded successfully.")
    
    # Load data for 1 batch
    print("Loading test data...")
    with open(PROCESSED_DIR / "user_positives.pkl", "rb") as f:
        user_positives = pickle.load(f)
    bpr_data = np.load(PROCESSED_DIR / "bpr_data.npz")
    all_items = bpr_data["all_items"]
    item_pop = bpr_data["item_pop_values"]
    
    sbert_data = torch.load(PROCESSED_DIR / "sbert_embeddings.pt", weights_only=True)
    sbert = sbert_data["embeddings"]
    popularity = torch.load(PROCESSED_DIR / "popularity.pt", weights_only=True)
    user_features = torch.load(PROCESSED_DIR / "user_features.pt", weights_only=True)
    history = torch.load(PROCESSED_DIR / "history_embeddings.pt", weights_only=True)
    genre_table = torch.load(PROCESSED_DIR / "genre_table.pt", weights_only=False)
    
    dataset = BPRDataset(
        user_positives, all_items, item_pop,
        sbert, popularity, user_features, history, genre_table,
        n_samples_per_epoch=100
    )
    
    loader = DataLoader(dataset, batch_size=10, collate_fn=collate_bpr)
    batch = next(iter(loader))
    
    (uids, pos_ids, neg_ids, sbert_pos, sbert_neg,
     pop_pos, pop_neg, ufeats, hists, genre_pos, genre_neg) = batch
     
    # Move to device
    uids, pos_ids, neg_ids = uids.to(DEVICE), pos_ids.to(DEVICE), neg_ids.to(DEVICE)
    sbert_pos, sbert_neg = sbert_pos.to(DEVICE), sbert_neg.to(DEVICE)
    pop_pos, pop_neg = pop_pos.to(DEVICE), pop_neg.to(DEVICE)
    ufeats, hists = ufeats.to(DEVICE), hists.to(DEVICE)
    
    print("Running forward pass...")
    with torch.no_grad():
        score_pos, score_neg = model.predict_pair_scores(
            uids, pos_ids, neg_ids,
            sbert_pos, sbert_neg, pop_pos, pop_neg,
            genre_pos, genre_neg, hists, ufeats
        )
    
    print(f"Positive scores: {score_pos.flatten().cpu().numpy()}")
    print(f"Negative scores: {score_neg.flatten().cpu().numpy()}")
    
    diff = score_pos - score_neg
    print(f"Mean score diff: {diff.mean().item():.4f}")
    
    if (diff > 0).any():
        print("VERIFICATION SUCCESSFUL: Model produces ranking scores.")
    else:
        print("VERIFICATION FAILED: Model produced identical scores or failed.")

if __name__ == "__main__":
    verify_path_b()
