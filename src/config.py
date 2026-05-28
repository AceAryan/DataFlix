"""
DataFlix — Central Configuration
src/config.py
"""

import torch
from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR   = ROOT_DIR / "results"

# ── Raw dataset paths ─────────────────────────────────────────────────────────
ML_RAW_DIR   = RAW_DIR / "ml-32m"
IMDB_RAW_DIR = RAW_DIR / "imdb"
TMDB_RAW_DIR = RAW_DIR / "tmdb"

ML_RATINGS_PATH = ML_RAW_DIR / "ratings.csv"
ML_MOVIES_PATH  = ML_RAW_DIR / "movies.csv"
ML_LINKS_PATH   = ML_RAW_DIR / "links.csv"
ML_TAGS_PATH    = ML_RAW_DIR / "tags.csv"

IMDB_BASICS_PATH  = IMDB_RAW_DIR / "title.basics.tsv"
IMDB_RATINGS_PATH = IMDB_RAW_DIR / "title.ratings.tsv"
TMDB_CSV_PATH     = TMDB_RAW_DIR / "TMDB_movie_dataset_v11.csv"

# ── Processed paths ───────────────────────────────────────────────────────────
TRAIN_CSV           = PROCESSED_DIR / "train.csv"
VAL_CSV             = PROCESSED_DIR / "val.csv"
TEST_CSV            = PROCESSED_DIR / "test.csv"
USER_MAP_CSV        = PROCESSED_DIR / "user_map.csv"
MOVIE_MAP_CSV       = PROCESSED_DIR / "movie_map.csv"
COLD_START_CSV      = PROCESSED_DIR / "cold_start_users.csv"
STATS_JSON          = PROCESSED_DIR / "stats.json"
CSR_MATRIX_PATH     = PROCESSED_DIR / "train_csr.npz"
BPR_DATA_PATH       = PROCESSED_DIR / "bpr_data.npz"
USER_POSITIVES_PATH = PROCESSED_DIR / "user_positives.pkl"

SBERT_EMBEDDINGS_PATH   = PROCESSED_DIR / "sbert_embeddings.pt"
IMDB_FEATURES_PATH      = PROCESSED_DIR / "imdb_features.pt"
POPULARITY_PATH         = PROCESSED_DIR / "popularity.pt"
HISTORY_EMBEDDINGS_PATH = PROCESSED_DIR / "history_embeddings.pt"
GENRE_TABLE_PATH        = PROCESSED_DIR / "genre_table.pt"

ALS_PATH         = RESULTS_DIR / "als_factors.npz"
BPR_FACTORS_PATH = RESULTS_DIR / "bpr_factors.npz"
HYBRID_CKPT_PATH = RESULTS_DIR / "hybrid_best.pt"

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _configure_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.cuda.set_per_process_memory_fraction(0.85)

_configure_cuda()

def _detect_profile() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 14:   return "high_vram"
    elif vram_gb >= 7:  return "mid_vram"
    else:               return "low_vram"

_PROFILES = {
    "high_vram": dict(batch=8192, bpr_b=8192, als=20, bpr=50, samp=500_000, ep=100, eval=4096),
    "mid_vram":  dict(batch=6144, bpr_b=6144, als=15, bpr=50, samp=300_000, ep=75,  eval=2048),
    "low_vram":  dict(batch=4096, bpr_b=4096, als=12, bpr=50, samp=200_000, ep=50,  eval=1024),
    "cpu":       dict(batch=2048, bpr_b=2048, als=10, bpr=20, samp=50_000,  ep=30,  eval=512),
}
DEVICE_PROFILE = _detect_profile()
_P             = _PROFILES[DEVICE_PROFILE]

# ── Preprocessing ─────────────────────────────────────────────────────────────
MIN_USER_RATINGS     = 10   # k-core: drop users with fewer ratings
MIN_MOVIE_RATINGS    = 10   # k-core: drop movies with fewer ratings
COLD_START_THRESHOLD = 20   # users with < this many train ratings are "cold"
TRAIN_RATIO          = 0.8
VAL_RATIO            = 0.1
TEST_RATIO           = 0.1

# ── Feature dims ──────────────────────────────────────────────────────────────
SBERT_DIM     = 384   # all-MiniLM-L6-v2
IMDB_FEAT_DIM = 23    # 20 genre OHE + runtime + avg_vote + log_num_votes
NUM_GENRES    = 20

# ── Model hyperparams ─────────────────────────────────────────────────────────
LATENT_DIM_K = 128       # latent dimension for ALS and BPR
EMBED_DIM_D  = 256       # common projection dim for Hybrid
NUM_HEADS    = 4         # attention heads in Hybrid
MLP_HIDDEN   = [512, 128]
DROPOUT      = 0.2

# ── ALS ───────────────────────────────────────────────────────────────────────
ALS_ITERATIONS      = _P["als"]
ALS_REG             = 0.1
ALS_CONVERGENCE_TOL = 1e-4

# ── BPR ───────────────────────────────────────────────────────────────────────
LR_BPR              = 1e-3
BPR_REG             = 1e-4
BPR_EPOCHS          = _P["bpr"]
BPR_BATCH_SIZE      = _P["bpr_b"]
BPR_SAMPLES_PER_EPOCH = _P["samp"]

# ── Hybrid ────────────────────────────────────────────────────────────────────
LR_HYBRID           = 1e-3
HYBRID_WEIGHT_DECAY = 1e-4
HYBRID_EPOCHS       = _P["ep"]
HYBRID_BATCH_SIZE   = _P["batch"]
HYBRID_SAMPLES_PER_EPOCH = _P["samp"]
EARLY_STOP_PATIENCE = 7
COSINE_T_MAX        = 30
FREEZE_EPOCHS       = 5

# ── Evaluation ────────────────────────────────────────────────────────────────
TOP_K_VALUES    = [5, 10, 20]
EVAL_BATCH_SIZE = _P["eval"]
RELEVANCE_RATING = 4.0   # items rated >= this are "liked" for ranking eval

# ── Optuna ────────────────────────────────────────────────────────────────────
OPTUNA_N_TRIALS    = 50
OPTUNA_LATENT_DIMS = [64, 128, 256]
OPTUNA_REG_RANGE   = (1e-4, 1e-1)
OPTUNA_LR_RANGE    = (1e-4, 1e-2)

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

import logging as _log
_log.getLogger(__name__).info(
    f"Profile={DEVICE_PROFILE} | device={DEVICE} | "
    f"batch={HYBRID_BATCH_SIZE} | bpr_ep={BPR_EPOCHS} | als_iter={ALS_ITERATIONS}"
)