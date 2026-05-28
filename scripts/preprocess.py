"""
DataFlix — Preprocessing Pipeline
scripts/run_preprocessing.py

Usage:
  python scripts/run_preprocessing.py                # full pipeline
  python scripts/run_preprocessing.py --skip-sbert   # skip SBERT re-encoding
  python scripts/run_preprocessing.py --from-step 2  # skip parsing
  python scripts/run_preprocessing.py --from-step 3  # features only
"""

import argparse, json, logging, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.config import (
    PROCESSED_DIR, STATS_JSON,
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    MOVIE_MAP_CSV, USER_MAP_CSV,
    CSR_MATRIX_PATH, SBERT_EMBEDDINGS_PATH, IMDB_FEATURES_PATH,
    set_seed,
)
from src.data.parse      import load_ratings
from src.data.preprocess import run_preprocessing
from src.data.features   import run_feature_engineering

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _elapsed(t): s = time.time()-t; return f"{int(s//60)}m {s%60:.1f}s"

def _banner(title):
    log.info("")
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


def run_pipeline(from_step: int = 1, skip_sbert: bool = False) -> None:
    t0 = time.time()
    set_seed()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║   DataFlix Preprocessing  —  ML-32M + IMDB + TMDB   ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    # Step 1: Parse
    ratings = None
    if from_step <= 1:
        _banner("STEP 1 — Parse & Validate Ratings")
        t = time.time()
        ratings = load_ratings()
        log.info(f"  Done ({_elapsed(t)})")
    else:
        log.info("Skipping Step 1")

    # Step 2: Preprocess
    stats = {}
    if from_step <= 2:
        _banner("STEP 2 — Filter · Split · Map IDs · CSR · BPR")
        t = time.time()
        if ratings is None:
            ratings = load_ratings()
        stats = run_preprocessing(ratings)
        log.info(f"  Done ({_elapsed(t)})")
    else:
        log.info("Skipping Step 2")
        if not CSR_MATRIX_PATH.exists():
            raise FileNotFoundError("CSR matrix not found. Run from step 2.")
        if STATS_JSON.exists():
            with open(STATS_JSON) as f:
                stats = json.load(f)

    # Step 3: Features
    if from_step <= 3:
        _banner("STEP 3 — IMDB Features · SBERT · Popularity · History")
        t = time.time()
        movie_map_df = pd.read_csv(MOVIE_MAP_CSV)
        user_map_df  = pd.read_csv(USER_MAP_CSV)
        train_df     = pd.read_csv(TRAIN_CSV)
        movie_map = dict(zip(movie_map_df["movie_id"], movie_map_df["movie_idx"]))
        user_map  = dict(zip(user_map_df["user_id"],   user_map_df["user_idx"]))
        run_feature_engineering(
            movie_map=movie_map, user_map=user_map,
            train_df=train_df, skip_sbert=skip_sbert,
        )
        log.info(f"  Done ({_elapsed(t)})")
    else:
        log.info("Skipping Step 3")

    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  PREPROCESSING COMPLETE                               ║")
    log.info(f"║  Total: {_elapsed(t0):<47}║")
    log.info("╚══════════════════════════════════════════════════════╝")
    if stats:
        log.info(f"  Users={stats.get('n_users',0):,} | Movies={stats.get('n_movies',0):,} | "
                 f"Train={stats.get('n_train',0):,} | Density={stats.get('density',0):.4f}%")

    # Output checklist
    from src.config import (BPR_DATA_PATH, USER_POSITIVES_PATH, COLD_START_CSV,
                             POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH, GENRE_TABLE_PATH)
    files = [
        ("Train",        TRAIN_CSV),
        ("Val",          VAL_CSV),
        ("Test",         TEST_CSV),
        ("User map",     USER_MAP_CSV),
        ("Movie map",    MOVIE_MAP_CSV),
        ("CSR matrix",   CSR_MATRIX_PATH),
        ("BPR data",     BPR_DATA_PATH),
        ("User pos",     USER_POSITIVES_PATH),
        ("Cold users",   COLD_START_CSV),
        ("IMDB feats",   IMDB_FEATURES_PATH),
        ("SBERT embs",   SBERT_EMBEDDINGS_PATH),
        ("Popularity",   POPULARITY_PATH),
        ("History embs", HISTORY_EMBEDDINGS_PATH),
        ("Genre table",  GENRE_TABLE_PATH),
        ("Stats",        STATS_JSON),
    ]
    for label, path in files:
        ok   = "✓" if path.exists() else "✗"
        size = f"  ({path.stat().st_size/1e6:.1f}MB)" if path.exists() else ""
        log.info(f"  [{ok}] {label:<16} {path.name}{size}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--from-step", type=int, default=1, choices=[1,2,3])
    p.add_argument("--skip-sbert", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(from_step=args.from_step, skip_sbert=args.skip_sbert)