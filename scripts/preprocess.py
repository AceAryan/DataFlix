"""
DataFlix — Master Preprocessing Pipeline
scripts/run_preprocessing.py

Data sources:
  MovieLens 25M  — ratings, movies, links (training + evaluation)
  IMDB           — title.basics.tsv + title.ratings.tsv (content features)
  TMDB           — TMDB_movie_dataset_v11.csv (SBERT synopses)

Netflix Prize was evaluated as a cross-domain validation source but dropped:
  - Title/year alignment achieved only 5.9% movie coverage (1,045 / 17,770)
  - 94.1% of Netflix ratings mapped to movies outside the shared item space
  - Cross-domain evaluation was statistically unreliable
  - Decision documented in parse.py and preprocess.py

Usage:
  python scripts/run_preprocessing.py                  # Full pipeline
  python scripts/run_preprocessing.py --skip-sbert     # Skip SBERT re-encoding
  python scripts/run_preprocessing.py --from-step 2    # Skip alignment
  python scripts/run_preprocessing.py --from-step 3    # Re-run features only
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import (
    PROCESSED_DIR, STATS_JSON,
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    MOVIE_MAP_CSV, USER_MAP_CSV,
    CSR_MATRIX_PATH, SBERT_EMBEDDINGS_PATH,
    IMDB_FEATURES_PATH,
    set_seed,
)
from src.data.parse      import load_all_ratings
from src.data.preprocess import run_preprocessing
from src.data.features   import run_feature_engineering

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _step2_done() -> bool:
    return all(p.exists() for p in [TRAIN_CSV, VAL_CSV, TEST_CSV,
                                     MOVIE_MAP_CSV, USER_MAP_CSV,
                                     CSR_MATRIX_PATH])

def _step3_done(skip_sbert: bool) -> bool:
    required = [IMDB_FEATURES_PATH]
    if not skip_sbert:
        required.append(SBERT_EMBEDDINGS_PATH)
    return all(p.exists() for p in required)

def _banner(step: int, title: str) -> None:
    log.info("")
    log.info("=" * 60)
    log.info(f"  STEP {step}: {title}")
    log.info("=" * 60)

def _elapsed(start: float) -> str:
    s = time.time() - start
    return f"{int(s//60)}m {s%60:.1f}s"


def run_pipeline(from_step: int = 1, skip_sbert: bool = False) -> None:
    t_total = time.time()
    set_seed()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║        DataFlix Preprocessing Pipeline               ║")
    log.info("║        Source: MovieLens 25M + IMDB + TMDB           ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    # ── Step 1: Parse ─────────────────────────────────────────────────────────
    if from_step <= 1:
        _banner(1, "Parse & Validate Ratings")
        t = time.time()
        ratings = load_all_ratings()
        log.info(f"  Step 1 complete ({_elapsed(t)}) — {len(ratings):,} ratings loaded")
    else:
        log.info("Skipping Step 1 (--from-step >= 2)")
        ratings = None

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    if from_step <= 2:
        _banner(2, "Filter · Split · Map IDs · CSR · BPR")
        t = time.time()
        if ratings is None:
            log.info("  Re-parsing ratings for step 2...")
            ratings = load_all_ratings()
        stats = run_preprocessing(ratings)
        log.info(f"  Step 2 complete ({_elapsed(t)})")
        _print_stats(stats)
    else:
        log.info("Skipping Step 2 (--from-step >= 3)")
        if not _step2_done():
            raise FileNotFoundError("Processed CSVs not found. Run from step 2.")
        if STATS_JSON.exists():
            with open(STATS_JSON) as f:
                stats = json.load(f)

    # ── Step 3: Feature engineering ───────────────────────────────────────────
    if from_step <= 3:
        _banner(3, "IMDB Features · SBERT Embeddings · Popularity · History")
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
        log.info(f"  Step 3 complete ({_elapsed(t)})")
    else:
        log.info("Skipping Step 3 (--from-step >= 4)")

    # ── Done ──────────────────────────────────────────────────────────────────
    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  PREPROCESSING COMPLETE                               ║")
    log.info(f"║  Total time: {_elapsed(t_total):<42}║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info("")
    _list_outputs()


def _print_stats(stats: dict) -> None:
    log.info("")
    log.info("  Dataset statistics:")
    log.info(f"    Users          : {stats['n_users']:>10,}")
    log.info(f"    Movies         : {stats['n_movies']:>10,}")
    log.info(f"    Train ratings  : {stats['n_train']:>10,}")
    log.info(f"    Val ratings    : {stats['n_val']:>10,}")
    log.info(f"    Test ratings   : {stats['n_test']:>10,}")
    log.info(f"    Cold-start     : {stats['n_cold_start']:>10,}")
    log.info(f"    Matrix density : {stats['density_pct']:>9.4f}%")


def _list_outputs() -> None:
    from src.config import (
        NF_TO_ML_MAP_JSON, BPR_DATA_PATH, USER_POSITIVES_PATH,
        COLD_START_CSV, POPULARITY_PATH, HISTORY_EMBEDDINGS_PATH,
        GENRE_TABLE_PATH, MOVIE_MAP_CSV, USER_MAP_CSV,
    )
    output_files = [
        ("Train split",          TRAIN_CSV),
        ("Val split",            VAL_CSV),
        ("Test split",           TEST_CSV),
        ("User map",             USER_MAP_CSV),
        ("Movie map",            MOVIE_MAP_CSV),
        ("CSR matrix",           CSR_MATRIX_PATH),
        ("BPR data",             BPR_DATA_PATH),
        ("User positives",       USER_POSITIVES_PATH),
        ("Cold-start users",     COLD_START_CSV),
        ("IMDB features",        IMDB_FEATURES_PATH),
        ("SBERT embeddings",     SBERT_EMBEDDINGS_PATH),
        ("Popularity",           POPULARITY_PATH),
        ("History embeddings",   PROCESSED_DIR / "history_embeddings.pt"),
        ("Genre table",          GENRE_TABLE_PATH),
        ("Stats",                STATS_JSON),
    ]
    for label, path in output_files:
        status = "✓" if path.exists() else "✗ MISSING"
        size   = f"  ({path.stat().st_size / 1e6:.1f} MB)" if path.exists() else ""
        log.info(f"  [{status}] {label:<22} {path.name}{size}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DataFlix preprocessing pipeline (MovieLens 25M)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_preprocessing.py                Full pipeline
  python scripts/run_preprocessing.py --skip-sbert   Skip SBERT re-encoding
  python scripts/run_preprocessing.py --from-step 2  Skip parsing
  python scripts/run_preprocessing.py --from-step 3  Re-run features only
        """,
    )
    parser.add_argument("--from-step", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--skip-sbert", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(from_step=args.from_step, skip_sbert=args.skip_sbert)