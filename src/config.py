"""
DataFlix — Central Configuration
All hyperparameters, paths, and device selection.
"""

import torch
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"

NETFLIX_RAW_DIR = RAW_DIR / "netflix"
NETFLIX_ARCHIVE_DIR = ROOT_DIR / "archive"  # Alternative location
MOVIELENS_RAW_DIR = RAW_DIR / "movielens"
TMDB_RAW_DIR = RAW_DIR / "tmdb"

# Local dataset files (no API needed)
TMDB_CSV_PATH = ROOT_DIR / "tmdb" / "TMDB_movie_dataset_v11.csv"
IMDB_BASICS_PATH = ROOT_DIR / "imdb" / "imdb tsv" / "title.basics.tsv"
IMDB_RATINGS_PATH = ROOT_DIR / "imdb" / "imdb tsv" / "title.ratings.tsv"
ML_LINKS_PATH = ROOT_DIR / "ml-25m" / "ml-25m" / "links.csv"

# Processed file paths
RATINGS_CSV = PROCESSED_DIR / "ratings.csv"
MOVIES_CSV = PROCESSED_DIR / "movies_metadata.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
COLD_START_CSV = PROCESSED_DIR / "cold_start_users.csv"
CSR_MATRIX_PATH = PROCESSED_DIR / "train_csr.npz"
BPR_DATA_PATH = PROCESSED_DIR / "bpr_data.npz"

SBERT_EMBEDDINGS_PATH = PROCESSED_DIR / "sbert_embeddings.pt"
GENRE_TABLE_PATH = PROCESSED_DIR / "genre_table.pt"
USER_FEATURES_PATH = PROCESSED_DIR / "user_features.pt"
POPULARITY_PATH = PROCESSED_DIR / "popularity.pt"
HISTORY_EMBEDDINGS_PATH = PROCESSED_DIR / "history_embeddings.pt"

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Data Preprocessing
# ──────────────────────────────────────────────
MIN_USER_RATINGS = 5       # Remove users with fewer ratings
MIN_MOVIE_RATINGS = 10     # Remove movies with fewer ratings
TRAIN_RATIO = 0.8          # Temporal split ratios
VAL_RATIO = 0.1
TEST_RATIO = 0.1
COLD_START_THRESHOLD = 3   # Users with < this many train ratings

# ──────────────────────────────────────────────
# Model Hyperparameters (defaults; tuned by Optuna)
# ──────────────────────────────────────────────
LATENT_DIM_K = 100         # MF latent dimension k
EMBED_DIM_D = 128          # Common projection dimension d
NUM_HEADS = 4              # Self-attention heads H
MLP_HIDDEN = [256, 64]     # MLP prediction head
DROPOUT = 0.2
NUM_GENRES = 20            # Genre embedding table size
SBERT_DIM = 384            # SBERT output dimension (all-MiniLM-L6-v2)

# ──────────────────────────────────────────────
# Training — Path A (MSE)
# ──────────────────────────────────────────────
LR_PATH_A = 1e-3   # was 5e-3 — too aggressive, causes overshooting           # η₀
WEIGHT_DECAY = 1e-4         # λ (regularisation)
COSINE_T_MAX = 50           # Cosine annealing T_max
EARLY_STOP_PATIENCE = 5
MAX_EPOCHS = 100
BATCH_SIZE = 8192

# ──────────────────────────────────────────────
# Training — Path B (BPR)
# ──────────────────────────────────────────────
LR_PATH_B = 1e-3
BPR_REG = 1e-4
BPR_EPOCHS = 50
BPR_BATCH_SIZE = 8192
BPR_SAMPLES_PER_EPOCH = 200_000 

# ──────────────────────────────────────────────
# ALS
# ──────────────────────────────────────────────
ALS_ITERATIONS = 20
ALS_REG = 0.1
ALS_CONVERGENCE_TOL = 1e-4
ALS_PATH = RESULTS_DIR/"als_factors.npz"

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
TOP_K_VALUES = [5, 10, 20]

# ──────────────────────────────────────────────
# Optuna HPO
# ──────────────────────────────────────────────
OPTUNA_N_TRIALS = 50
OPTUNA_LATENT_DIMS = [50, 100, 200]
OPTUNA_REG_RANGE = (1e-4, 1e-1)
OPTUNA_LR_RANGE = (1e-4, 1e-2)
OPTUNA_HEADS = [2, 4, 8]
OPTUNA_EMBED_DIMS = [64, 128, 256]

# External APIs — NOT USED (local datasets used instead)
# TMDB_API_KEY = "..."  # API not needed; use tmdb/TMDB_movie_dataset_v11.csv

# ──────────────────────────────────────────────
# Random seed
# ──────────────────────────────────────────────
SEED = 42

def set_seed(seed=SEED):
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
