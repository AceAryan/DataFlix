"""
DataFlix — Central Configuration
All hyperparameters, paths, and device selection.
"""

import torch
from pathlib import Path

# ──────────────────────────────────────────────
# Root & top-level dirs
# ──────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"

# ──────────────────────────────────────────────
# Raw dataset dirs  (all under data/raw/)
# ──────────────────────────────────────────────
NETFLIX_RAW_DIR   = RAW_DIR / "netflix"
MOVIELENS_RAW_DIR = RAW_DIR / "ml-25m" 
TMDB_RAW_DIR      = RAW_DIR / "tmdb"
IMDB_RAW_DIR      = RAW_DIR / "imdb"

# ──────────────────────────────────────────────
# Raw file paths
# ──────────────────────────────────────────────
# MovieLens
ML_RATINGS_PATH = MOVIELENS_RAW_DIR / "ratings.csv"
ML_MOVIES_PATH  = MOVIELENS_RAW_DIR / "movies.csv"
ML_LINKS_PATH   = MOVIELENS_RAW_DIR / "links.csv"

# Netflix  (pre-cleaned CSVs)
NETFLIX_MOVIES_PATH  = NETFLIX_RAW_DIR / "Netflix_Dataset_Movie.csv"
NETFLIX_RATINGS_PATH = NETFLIX_RAW_DIR / "Netflix_Dataset_Rating.csv"

# TMDB  (local CSV)
TMDB_CSV_PATH = TMDB_RAW_DIR / "TMDB_movie_dataset_v11.csv"

# IMDB  (TSV dumps from datasets.imdbws.com)
IMDB_BASICS_PATH  = IMDB_RAW_DIR / "title.basics.tsv"
IMDB_RATINGS_PATH = IMDB_RAW_DIR / "title.ratings.tsv"

# ──────────────────────────────────────────────
# Processed file paths
# ──────────────────────────────────────────────
# Ratings splits
TRAIN_CSV      = PROCESSED_DIR / "train.csv"
VAL_CSV        = PROCESSED_DIR / "val.csv"
TEST_CSV       = PROCESSED_DIR / "test.csv"
COLD_START_CSV = PROCESSED_DIR / "cold_start_users.csv"

# ID mappings
USER_MAP_CSV  = PROCESSED_DIR / "user_map.csv"
MOVIE_MAP_CSV = PROCESSED_DIR / "movie_map.csv"

# Metadata
MOVIES_META_CSV = PROCESSED_DIR / "movies_metadata.csv"   # ML genres + IMDB enrichment
STATS_JSON      = PROCESSED_DIR / "stats.json"

# Alignment map
NF_TO_ML_MAP_JSON = PROCESSED_DIR / "netflix_to_ml_movie_map.json"

# CF arrays
CSR_MATRIX_PATH = PROCESSED_DIR / "train_csr.npz"
BPR_DATA_PATH   = PROCESSED_DIR / "bpr_data.npz"
USER_POSITIVES_PATH = PROCESSED_DIR / "user_positives.pkl"

# Feature tensors
SBERT_EMBEDDINGS_PATH   = PROCESSED_DIR / "sbert_embeddings.pt"
GENRE_TABLE_PATH        = PROCESSED_DIR / "genre_table.pt"
USER_FEATURES_PATH      = PROCESSED_DIR / "user_features.pt"
POPULARITY_PATH         = PROCESSED_DIR / "popularity.pt"
HISTORY_EMBEDDINGS_PATH = PROCESSED_DIR / "history_embeddings.pt"
IMDB_FEATURES_PATH      = PROCESSED_DIR / "imdb_features.pt"   # genre OHE + runtime + vote stats

# ALS output
ALS_PATH = RESULTS_DIR / "als_factors.npz"

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Data preprocessing thresholds
# ──────────────────────────────────────────────
MIN_USER_RATINGS  = 20   # Discard users with fewer ratings
MIN_MOVIE_RATINGS = 10   # Discard movies with fewer ratings
COLD_START_THRESHOLD = 5 # Users with < this many train ratings are "cold"

# Per-user temporal split ratios  (applied independently to ML and Netflix users)
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

# ──────────────────────────────────────────────
# Model hyperparameters  (defaults; tuned by Optuna)
# ──────────────────────────────────────────────
LATENT_DIM_K = 100          # MF latent dimension k
EMBED_DIM_D  = 128          # Common projection dimension d
NUM_HEADS    = 4            # Self-attention heads H
MLP_HIDDEN   = [256, 64]    # MLP prediction head layer widths
DROPOUT      = 0.2
NUM_GENRES   = 20           # Genre embedding table size
SBERT_DIM    = 384          # all-MiniLM-L6-v2 output dimension
IMDB_FEAT_DIM = 23          # genre OHE (20) + runtime (1) + avg_vote (1) + num_votes_log (1)

# ──────────────────────────────────────────────
# Training — Path A  (MSE / rating prediction)
# ──────────────────────────────────────────────
LR_PATH_A          = 1e-3   # η₀ initial learning rate
WEIGHT_DECAY       = 1e-4   # λ regularisation
COSINE_T_MAX       = 50     # cosine annealing T_max
EARLY_STOP_PATIENCE = 5
MAX_EPOCHS         = 100
BATCH_SIZE         = 8192

# ──────────────────────────────────────────────
# Training — Path B  (BPR / ranking)
# ──────────────────────────────────────────────
LR_PATH_B           = 1e-3
BPR_REG             = 1e-4
BPR_EPOCHS          = 50
BPR_BATCH_SIZE      = 8192
BPR_SAMPLES_PER_EPOCH = 200_000

# ──────────────────────────────────────────────
# ALS
# ──────────────────────────────────────────────
ALS_ITERATIONS    = 20
ALS_REG           = 0.1
ALS_CONVERGENCE_TOL = 1e-4

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
TOP_K_VALUES = [5, 10, 20]   # Recall@K, NDCG@K, MRR@K

# ──────────────────────────────────────────────
# Optuna Hyperparameter Optimization
# ──────────────────────────────────────────────
OPTUNA_N_TRIALS    = 50
OPTUNA_LATENT_DIMS = [50, 100, 200]
OPTUNA_REG_RANGE   = (1e-4, 1e-1)
OPTUNA_LR_RANGE    = (1e-4, 1e-2)
OPTUNA_HEADS       = [2, 4, 8]
OPTUNA_EMBED_DIMS  = [64, 128, 256]

# ──────────────────────────────────────────────
# Random seed
# ──────────────────────────────────────────────
SEED = 13

def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False