"""
DataFlix — Master Preprocessing Pipeline
scripts/run_preprocessing.py

Wires together all four data modules in order:
  1. align.py     — Netflix → MovieLens movie ID alignment
  2. parse.py     — Load and clean ratings from both sources
  3. preprocess.py — k-core filter, temporal split, ID mapping, CSR/BPR arrays
  4. features.py  — IMDB features, SBERT embeddings, popularity, user history

Usage:
  python scripts/run_preprocessing.py                  # Full pipeline
  python scripts/run_preprocessing.py --skip-sbert     # Skip SBERT re-encoding
  python scripts/run_preprocessing.py --force-align    # Force rebuild alignment map
  python scripts/run_preprocessing.py --from-step 3    # Resume from a specific step

Steps are checkpointed — if processed files already exist, steps are skipped
unless --force flags are passed.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import (
    PROCESSED_DIR, STATS_JSON,
    NF_TO_ML_MAP_JSON,
    TRAIN_CSV, VAL_CSV, TEST_CSV,
    MOVIE_MAP_CSV, USER_MAP_CSV,
    CSR_MATRIX_PATH, SBERT_EMBEDDINGS_PATH,
    IMDB_FEATURES_PATH,
    set_seed,
)
from src.data.align      import run_alignment, run_alignment_force
from src.data.parse      import load_all_ratings
from src.data.preprocess import run_preprocessing
from src.data.features   import run_feature_engineering

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _step1_done() -> bool:
    return NF_TO_ML_MAP_JSON.exists()

def _step2_done() -> bool:
    return _step1_done()  # parse produces no files; depends on align

def _step3_done() -> bool:
    return all(p.exists() for p in [TRAIN_CSV, VAL_CSV, TEST_CSV,
                                     MOVIE_MAP_CSV, USER_MAP_CSV,
                                     CSR_MATRIX_PATH])

def _step4_done(skip_sbert: bool) -> bool:
    required = [IMDB_FEATURES_PATH]
    if not skip_sbert:
        required.append(SBERT_EMBEDDINGS_PATH)
    return all(p.exists() for p in required)


def _print_banner(step: int, title: str) -> None:
    log.info("")
    log.info("=" * 60)
    log.info(f"  STEP {step}: {title}")
    log.info("=" * 60)


def _elapsed(start: float) -> str:
    s = time.time() - start
    return f"{int(s//60)}m {s%60:.1f}s"


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    from_step:    int  = 1,
    force_align:  bool = False,
    skip_sbert:   bool = False,
) -> None:
    t_total = time.time()
    set_seed()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║        DataFlix Preprocessing Pipeline               ║")
    log.info("║  Train: MovieLens 25M + Netflix  |  Eval: held-out   ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    # ── Step 1: Alignment ─────────────────────────────────────────────────────
    if from_step <= 1:
        _print_banner(1, "Netflix → MovieLens Alignment")
        t = time.time()

        if force_align:
            log.info("  --force-align set: rebuilding alignment map from scratch")
            nf_to_ml_map = run_alignment_force()
        elif _step1_done():
            log.info(f"  Checkpoint found: {NF_TO_ML_MAP_JSON}")
            log.info("  Loading existing alignment map (use --force-align to rebuild)")
            with open(NF_TO_ML_MAP_JSON) as f:
                nf_to_ml_map = {int(k): v for k, v in json.load(f).items()}
        else:
            nf_to_ml_map = run_alignment()

        log.info(f"  Step 1 complete ({_elapsed(t)}) — "
                 f"{len(nf_to_ml_map):,} Netflix movies aligned")
    else:
        log.info("Skipping Step 1 (--from-step >= 2)")
        if not _step1_done():
            raise FileNotFoundError(
                f"Cannot skip Step 1 — alignment map not found at {NF_TO_ML_MAP_JSON}"
            )
        with open(NF_TO_ML_MAP_JSON) as f:
            nf_to_ml_map = {int(k): v for k, v in json.load(f).items()}

    # ── Step 2: Parse ratings ─────────────────────────────────────────────────
    if from_step <= 2:
        _print_banner(2, "Parse & Validate Ratings")
        t = time.time()
        ml_ratings, nf_ratings = load_all_ratings(nf_to_ml_map)
        log.info(f"  Step 2 complete ({_elapsed(t)}) — "
                 f"ML {len(ml_ratings):,} | NF {len(nf_ratings):,} ratings loaded")
    else:
        log.info("Skipping Step 2 (--from-step >= 3) — ratings will be loaded from CSV")
        ml_ratings = nf_ratings = None  # Loaded from disk in Step 3 if needed

    # ── Step 3: Preprocess ────────────────────────────────────────────────────
    if from_step <= 3:
        _print_banner(3, "Filter · Split · Map IDs · CSR · BPR")
        t = time.time()

        if ml_ratings is None or nf_ratings is None:
            # Resuming from step 3 — re-parse ratings (no checkpoint for raw DFs)
            log.info("  Re-parsing ratings for step 3...")
            ml_ratings, nf_ratings = load_all_ratings(nf_to_ml_map)

        stats = run_preprocessing(ml_ratings, nf_ratings)

        log.info(f"  Step 3 complete ({_elapsed(t)})")
        _print_stats(stats)
    else:
        log.info("Skipping Step 3 (--from-step >= 4)")
        if not _step3_done():
            raise FileNotFoundError(
                "Cannot skip Step 3 — processed CSVs not found. "
                "Run from step 3 or lower."
            )
        if STATS_JSON.exists():
            with open(STATS_JSON) as f:
                stats = json.load(f)

    # ── Step 4: Feature engineering ───────────────────────────────────────────
    if from_step <= 4:
        _print_banner(4, "IMDB Features · SBERT Embeddings · Popularity · History")
        t = time.time()

        # Load ID maps and training data from disk
        movie_map_df = pd.read_csv(MOVIE_MAP_CSV)
        user_map_df  = pd.read_csv(USER_MAP_CSV)
        train_df     = pd.read_csv(TRAIN_CSV)

        movie_map = dict(zip(movie_map_df["movie_id"], movie_map_df["movie_idx"]))
        user_map  = dict(zip(user_map_df["user_id"],   user_map_df["user_idx"]))

        feature_tensors = run_feature_engineering(
            movie_map   = movie_map,
            user_map    = user_map,
            train_df    = train_df,
            skip_sbert  = skip_sbert,
        )

        log.info(f"  Step 4 complete ({_elapsed(t)})")
    else:
        log.info("Skipping Step 4 (--from-step >= 5)")

    # ── Done ──────────────────────────────────────────────────────────────────
    log.info("")
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  PREPROCESSING COMPLETE                               ║")
    log.info(f"║  Total time: {_elapsed(t_total):<42}║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info("")
    log.info("Output files:")
    _list_outputs()


def _print_stats(stats: dict) -> None:
    log.info("")
    log.info("  Dataset statistics:")
    log.info(f"    Users          : {stats['n_users']:>10,}")
    log.info(f"    Movies         : {stats['n_movies']:>10,}")
    log.info(f"    Train ratings  : {stats['n_train']:>10,}  "
             f"(ML: {stats.get('ml_train',0):,} | NF: {stats.get('nf_train',0):,})")
    log.info(f"    Val ratings    : {stats['n_val']:>10,}  "
             f"(ML: {stats.get('ml_val',0):,} | NF: {stats.get('nf_val',0):,})")
    log.info(f"    Test ratings   : {stats['n_test']:>10,}  "
             f"(ML: {stats.get('ml_test',0):,} | NF: {stats.get('nf_test',0):,})")
    log.info(f"    Cold-start     : {stats['n_cold_start']:>10,}")
    log.info(f"    Matrix density : {stats['density_pct']:>9.4f}%")


def _list_outputs() -> None:
    output_files = [
        ("Alignment map",     NF_TO_ML_MAP_JSON),
        ("Alignment report",  PROCESSED_DIR / "alignment_report.csv"),
        ("Train split",       TRAIN_CSV),
        ("Val split",         VAL_CSV),
        ("Test split",        TEST_CSV),
        ("User map",          USER_MAP_CSV),
        ("Movie map",         MOVIE_MAP_CSV),
        ("CSR matrix",        CSR_MATRIX_PATH),
        ("BPR data",          PROCESSED_DIR / "bpr_data.npz"),
        ("User positives",    PROCESSED_DIR / "user_positives.pkl"),
        ("Cold-start users",  PROCESSED_DIR / "cold_start_users.csv"),
        ("IMDB features",     IMDB_FEATURES_PATH),
        ("SBERT embeddings",  SBERT_EMBEDDINGS_PATH),
        ("Popularity",        PROCESSED_DIR / "popularity.pt"),
        ("History embeddings",PROCESSED_DIR / "history_embeddings.pt"),
        ("Genre table",       PROCESSED_DIR / "genre_table.pt"),
        ("Stats",             STATS_JSON),
    ]
    for label, path in output_files:
        status = "✓" if path.exists() else "✗ MISSING"
        size   = f"  ({path.stat().st_size / 1e6:.1f} MB)" if path.exists() else ""
        log.info(f"  [{status}] {label:<22} {path.name}{size}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DataFlix end-to-end preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_preprocessing.py                   Full pipeline
  python scripts/run_preprocessing.py --skip-sbert      Skip SBERT re-encoding
  python scripts/run_preprocessing.py --force-align     Force alignment rebuild
  python scripts/run_preprocessing.py --from-step 3     Resume from filtering
  python scripts/run_preprocessing.py --from-step 4     Re-run features only
        """,
    )
    parser.add_argument(
        "--from-step", type=int, default=1, choices=[1, 2, 3, 4],
        help="Resume pipeline from this step (default: 1 = start from scratch)",
    )
    parser.add_argument(
        "--force-align", action="store_true",
        help="Force rebuild of Netflix→ML alignment map even if it exists",
    )
    parser.add_argument(
        "--skip-sbert", action="store_true",
        help="Skip SBERT encoding and use cached embeddings if available",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        from_step   = args.from_step,
        force_align = args.force_align,
        skip_sbert  = args.skip_sbert,
    )