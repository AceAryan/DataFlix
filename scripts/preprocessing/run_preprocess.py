"""
DataFlix — Cross-Domain Data Preprocessing Pipeline
End-to-end: parse ML25M & Netflix → filter → align → features.
Train on MovieLens, Validate on Netflix (Zero-Shot Cross-Domain).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from scipy import sparse

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import set_seed, PROCESSED_DIR, ROOT_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, CSR_MATRIX_PATH, BPR_DATA_PATH, COLD_START_CSV
from src.data.parse_netflix import parse_netflix
from src.data.preprocess import filter_sparse, create_id_mappings, mean_center_ratings, build_csr_matrix, binarise_for_bpr, identify_cold_start_users
from src.data.features import run_feature_engineering, load_synopses_from_tmdb, build_netflix_ml_alignment


def main():
    set_seed()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DataFlix — Cross-Domain Preprocessing Pipeline")
    print("Train: MovieLens 25M | Val/Test: Netflix")
    print("=" * 60)
    
    # --- Step 1: Build / Load aligned mapping ---
    mapping_path = PROCESSED_DIR / "netflix_to_ml_movie_map.json"
    if not mapping_path.exists():
        print("\n[STEP 1] Building Netflix -> MovieLens alignment map...")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # Build a temporary movie_map from ML movies to bootstrap alignment
        ml_movies_tmp = pd.read_csv(ROOT_DIR / "ml-25m" / "ml-25m" / "movies.csv")
        tmp_movie_map = dict(zip(ml_movies_tmp["movieId"], range(len(ml_movies_tmp))))
        build_netflix_ml_alignment(tmp_movie_map)
    with open(mapping_path) as f:
        nf_to_ml_map = {int(k): int(v) for k, v in json.load(f).items()}
    print(f"\n[STEP 1] Loaded mapping for {len(nf_to_ml_map):,} Netflix movies")
    
    # ─── Step 2: Load MovieLens (Train) ───
    print("\n[STEP 2] Load MovieLens 25M (Training Data)")
    ml_ratings = pd.read_csv(ROOT_DIR / "ml-25m" / "ml-25m" / "ratings.csv")
    ml_ratings = ml_ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    
    # Prefix user IDs to prevent collision
    ml_ratings["user_id"] = "ML_" + ml_ratings["user_id"].astype(str)
    
    # ─── Step 3: Load Netflix (Val/Test) ───
    print("\n[STEP 3] Load Netflix Data (Val/Test Data)")
    nf_ratings = parse_netflix()
    nf_ratings["user_id"] = "NF_" + nf_ratings["user_id"].astype(str)
    
    print("  Filtering Netflix ratings to aligned movies only...")
    nf_ratings = nf_ratings[nf_ratings["movie_id"].isin(nf_to_ml_map.keys())].copy()
    nf_ratings["movie_id"] = nf_ratings["movie_id"].map(nf_to_ml_map)
    print(f"  {len(nf_ratings):,} Netflix ratings retained on mapped movies.")
    
    # ─── Step 4: Filter & Map IDs ───
    print("\n[STEP 4] Filter Sparse and Map IDs")
    # Filter ML separately to preserve its dense core
    print("  Filtering MovieLens...")
    train_df = filter_sparse(ml_ratings)
    
    # Split Netflix into 50% val / 50% test by completely random split (since users are disjoint anyway)
    # The proposal says temporal split, but for zero-shot we just split users.
    print("  Splitting Netflix users into Val/Test...")
    val_users = np.random.choice(nf_ratings["user_id"].unique(), size=int(nf_ratings["user_id"].nunique()*0.5), replace=False)
    val_df = nf_ratings[nf_ratings["user_id"].isin(val_users)].copy()
    test_df = nf_ratings[~nf_ratings["user_id"].isin(val_users)].copy()
    
    
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # DROP NANS HERE to prevent IntCastingNaNError!
    combined = combined.dropna(subset=["user_id", "movie_id", "rating"])
    
    user_map, movie_map, combined_idx = create_id_mappings(combined)
    n_users, n_movies = len(user_map), len(movie_map)
    
    # Split back
    train_df = combined_idx.iloc[:len(train_df)].copy()
    val_df = combined_idx.iloc[len(train_df):len(train_df)+len(val_df)].copy()
    test_df = combined_idx.iloc[-len(test_df):].copy()
    
    pd.DataFrame(list(user_map.items()), columns=["user_id", "user_idx"]).to_csv(PROCESSED_DIR / "user_map.csv", index=False)
    pd.DataFrame(list(movie_map.items()), columns=["movie_id", "movie_idx"]).to_csv(PROCESSED_DIR / "movie_map.csv", index=False)
    
    # ─── Step 5: Mean Center ───
    print("\n[STEP 5] Mean Center Ratings")
    train_df = mean_center_ratings(train_df)
    val_df = mean_center_ratings(val_df)
    test_df = mean_center_ratings(test_df)
    
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    
    # ─── Step 6: GPU Arrays ───
    print("\n[STEP 6] GPU Arrays (CSR & BPR)")
    csr_mat = build_csr_matrix(train_df, n_users, n_movies)
    sparse.save_npz(CSR_MATRIX_PATH, csr_mat)
    
    user_pos, all_items, item_pop = binarise_for_bpr(train_df)
    np.savez(BPR_DATA_PATH, all_items=np.array(list(all_items)), item_pop_index=item_pop.index.values, item_pop_values=item_pop.values)
    import pickle
    with open(PROCESSED_DIR / "user_positives.pkl", "wb") as f:
        pickle.dump(user_pos, f)
        
    cold_users = identify_cold_start_users(train_df)
    pd.DataFrame({"user_idx": cold_users}).to_csv(COLD_START_CSV, index=False)
    
    # ─── Step 7: Prepare Features ───
    print("\n[STEP 7] Feature Engineering")
    ml_movies = pd.read_csv(ROOT_DIR / "ml-25m" / "ml-25m" / "movies.csv")
    if "movieId" in ml_movies.columns:
        ml_movies = ml_movies.rename(columns={"movieId": "movie_id"})
        
    ml_movies["movie_idx"] = ml_movies["movie_id"].map(movie_map)
    ml_movies = ml_movies.dropna(subset=["movie_idx"]).copy()
    ml_movies["movie_idx"] = ml_movies["movie_idx"].fillna(-1).astype(int)
    ml_movies.to_csv(PROCESSED_DIR / "movies_metadata.csv", index=False)
    
    synopses = load_synopses_from_tmdb(movie_map)
    run_feature_engineering(train_df, ml_movies, n_users, n_movies, synopses)
    
    stats = {
        "n_users": n_users, "n_movies": n_movies, "n_train": len(train_df),
        "n_val": len(val_df), "n_test": len(test_df), "n_cold_start": len(cold_users),
        "density_pct": float(csr_mat.nnz / (n_users * n_movies) * 100)
    }
    with open(PROCESSED_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nPREPROCESSING COMPLETE!")

if __name__ == "__main__":
    main()
