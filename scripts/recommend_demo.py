"""
DataFlix — Recommendation CLI
Usage:
    python recommend.py --user_id 42
    python recommend.py --user_id 42 --top_k 20
    python recommend.py --user_id 42 --mode a              # Path A only (rating prediction)
    python recommend.py --user_id 42 --mode b              # Path B only (ranking)
    python recommend.py --user_id 42 --mode ensemble       # 0.5*A + 0.5*B (default)
    python recommend.py --user_id 42 --genre Action
    python recommend.py --user_id 42 --exclude_watched false
    python recommend.py --user_id 42 --recency_boost 0.3   # favour newer movies
    python recommend.py --user_id 42 --recency_boost 0.0   # pure model scores (default)
    python recommend.py --cold_start                       # recommend for a brand-new user
    python recommend.py --stats                            # show dataset stats

recency_boost guide:
    0.0  -- pure model score, no year adjustment (default)
    0.1  -- light nudge toward recent movies
    0.3  -- moderate boost (recommended starting point)
    0.5  -- strong boost, post-2000 movies dominate
    1.0  -- very aggressive, only use with --genre filter
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import DEVICE, PROCESSED_DIR, RESULTS_DIR, ROOT_DIR
from src.models.hybrid import DataFlixModel
from src.training.trainer import init_feature_store, _build_genre_tensor

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_PATH      = PROCESSED_DIR / "train.csv"
MOVIES_PATH     = PROCESSED_DIR / "movies_metadata.csv"
STATS_PATH      = PROCESSED_DIR / "stats.json"
SBERT_PATH      = PROCESSED_DIR / "sbert_embeddings.pt"
HISTORY_PATH    = PROCESSED_DIR / "history_embeddings.pt"
POP_PATH        = PROCESSED_DIR / "popularity.pt"
USER_FEAT_PATH  = PROCESSED_DIR / "user_features.pt"
GENRE_PATH      = PROCESSED_DIR / "genre_table.pt"
MODEL_A_PATH    = RESULTS_DIR   / "dataflix_path_a.pt"
MODEL_B_PATH    = RESULTS_DIR   / "dataflix_path_b.pt"
USER_MAP_PATH   = PROCESSED_DIR / "user_map.csv"
MOVIE_MAP_PATH  = PROCESSED_DIR / "movie_map.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_assets():
    """Load all data assets once. Returns a dict of everything needed."""
    print("Loading assets...")
    t0 = time.time()

    # Stats
    with open(STATS_PATH) as f:
        stats = json.load(f)
    n_users  = stats["n_users"]
    n_movies = stats["n_movies"]

    # Interaction data
    train  = pd.read_csv(TRAIN_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    # Extract year from title: "Toy Story (1995)" -> 1995
    movies["_year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    movies["_year"] = movies["_year"].fillna(1990).clip(lower=1900, upper=2025)

    # Build movie index -> title/genre/year lookup
    idx2title = {}
    idx2genres_str = {}
    idx2year = {}
    for _, row in movies.iterrows():
        idx = int(row["movie_idx"])
        idx2title[idx]      = str(row.get("title", f"Movie {idx}"))
        idx2genres_str[idx] = str(row.get("genres", ""))
        idx2year[idx]       = float(row["_year"])

    # Build year tensor for recency boost: (n_movies,) normalised to [0, 1]
    # 1900 -> 0.0,  2025 -> 1.0
    year_arr = np.array([idx2year.get(i, 1990) for i in range(n_movies)], dtype=np.float32)
    year_norm = (year_arr - 1900.0) / (2025.0 - 1900.0)
    year_norm = np.clip(year_norm, 0.0, 1.0)
    movie_years = torch.tensor(year_norm, dtype=torch.float32)

    # Feature tensors — keep on CPU initially, init_feature_store moves to GPU
    sbert_data = torch.load(SBERT_PATH, weights_only=False)
    sbert      = sbert_data["embeddings"] if isinstance(sbert_data, dict) else sbert_data

    history   = torch.load(HISTORY_PATH,   weights_only=False)
    pop_raw   = torch.load(POP_PATH,       weights_only=False)
    user_feat = torch.load(USER_FEAT_PATH, weights_only=False)

    # Normalise popularity shape
    pop = pop_raw.float()
    if pop.dim() == 1:
        pop = pop.unsqueeze(1)   # (n_movies,) → (n_movies, 1)

    # Genre table — handle both old dict format and new tensor format
    genre_raw = torch.load(GENRE_PATH, weights_only=False)
    if torch.is_tensor(genre_raw):
        genre_tensor_cpu = genre_raw.long()
    elif isinstance(genre_raw, dict):
        genre_tensor_cpu = _build_genre_tensor(genre_raw, n_movies)
    else:
        raise ValueError(f"Unknown genre format: {type(genre_raw)}")

    # Build a fake genre_table dict for init_feature_store compatibility
    genre_table = {"movie_genre_ids": {}}   # already built above

    # Move features to GPU via init_feature_store
    # We pass tensors directly — rebuild what init_feature_store expects
    _init_store(sbert, pop.squeeze(1), user_feat, history, genre_tensor_cpu)

    # User history lookup for filtering watched items
    user_watched = train.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    # User map (original ID → internal idx) for display
    user_map_df = pd.read_csv(USER_MAP_PATH)
    orig2idx = dict(zip(user_map_df["user_id"], user_map_df["user_idx"]))
    idx2orig = dict(zip(user_map_df["user_idx"], user_map_df["user_id"]))

    print(f"  Assets loaded in {time.time()-t0:.1f}s  |  "
          f"Users: {n_users:,}  Movies: {n_movies:,}  Device: {DEVICE}")

    return {
        "n_users":        n_users,
        "n_movies":       n_movies,
        "train":          train,
        "idx2title":      idx2title,
        "idx2genres_str": idx2genres_str,
        "idx2year":       idx2year,
        "movie_years":    movie_years.to(DEVICE),  # (n_movies,) normalised year [0,1]
        "user_watched":   user_watched,
        "orig2idx":       orig2idx,
        "idx2orig":       idx2orig,
        "stats":          stats,
        "genre_tensor":   genre_tensor_cpu.to(DEVICE),
        "pop":            pop.to(DEVICE),
        "history":        history.to(DEVICE),
        "user_feat":      user_feat.to(DEVICE),
        "sbert":          sbert.to(DEVICE),
    }


def _init_store(sbert, pop_1d, user_feat, history, genre_tensor):
    """Directly populate trainer module globals without calling init_feature_store."""
    import src.training.trainer as T
    T._SBERT      = sbert.to(DEVICE)
    T._POPULARITY = pop_1d.to(DEVICE)
    T._USER_FEAT  = user_feat.to(DEVICE)
    T._HISTORY    = history.to(DEVICE)
    T._GENRE      = genre_tensor.to(DEVICE)


def load_models(n_users, n_movies):
    """Load Path A and Path B models."""
    models = {}

    for path_name, model_path in [("a", MODEL_A_PATH), ("b", MODEL_B_PATH)]:
        if not model_path.exists():
            print(f"  [warn] Model {model_path.name} not found — skipping Path {path_name.upper()}")
            continue
        m = DataFlixModel(n_users, n_movies, path=path_name.upper()).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE, weights_only=False)
        m.load_state_dict(state)
        m.eval()
        models[path_name] = m
        print(f"  Loaded Path {path_name.upper()} from {model_path.name}")

    if not models:
        raise RuntimeError("No trained models found in results/. Train first with run_train.py")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_all_items(user_idx: int, assets: dict, models: dict, mode: str) -> torch.Tensor:
    """
    Score all n_movies items for a given user in one vectorised forward pass.
    Returns (n_movies,) float tensor of scores.
    """
    n_movies = assets["n_movies"]

    user_t = torch.full((n_movies,), user_idx, dtype=torch.long, device=DEVICE)
    item_t = torch.arange(n_movies, device=DEVICE)

    # Batch-lookup features from GPU store
    sbert_t  = assets["sbert"][item_t]
    pop_t    = assets["pop"][item_t]
    genre_t  = assets["genre_tensor"][item_t]
    hist_t   = assets["history"][user_idx].unsqueeze(0).expand(n_movies, -1)
    ufeat_t  = assets["user_feat"][user_idx].unsqueeze(0).expand(n_movies, -1)

    scores_a = scores_b = None

    if "a" in models and mode in ("a", "ensemble"):
        with torch.amp.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            scores_a = models["a"](user_t, item_t, sbert_t, pop_t,
                                   genre_t, hist_t, ufeat_t)

    if "b" in models and mode in ("b", "ensemble"):
        with torch.amp.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            scores_b = models["b"](user_t, item_t, sbert_t, pop_t,
                                   genre_t, hist_t, ufeat_t)

    if mode == "ensemble" and scores_a is not None and scores_b is not None:
        # Normalise each to [0,1] before combining so scales match
        def norm01(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-8)
        scores = 0.5 * norm01(scores_a) + 0.5 * norm01(scores_b)
    elif scores_a is not None:
        scores = scores_a
    elif scores_b is not None:
        scores = scores_b
    else:
        raise RuntimeError(f"Mode '{mode}' requested but required model not loaded.")

    return scores.float()


def get_recommendations(user_idx: int, assets: dict, models: dict,
                        top_k: int = 10, mode: str = "ensemble",
                        exclude_watched: bool = True,
                        genre_filter: str = None,
                        recency_boost: float = 0.0) -> list[dict]:
    """
    Full recommendation pipeline for one user.
    Returns list of dicts with title, score, genres, year.

    recency_boost: 0.0 = pure model scores, 0.3 = moderate nudge toward newer movies.
    The boost is added as: final_score = norm(model_score) + recency_boost * year_norm
    where year_norm is 0.0 for 1900 and 1.0 for 2025.
    """
    scores = score_all_items(user_idx, assets, models, mode)

    # Normalise model scores to [0, 1] so recency boost is on the same scale
    lo, hi = scores.min(), scores.max()
    scores_norm = (scores - lo) / (hi - lo + 1e-8)

    # Recency boost — add a year-based bonus before masking
    if recency_boost > 0.0:
        year_norm = assets["movie_years"]   # (n_movies,) in [0, 1], already on GPU
        scores_norm = scores_norm + recency_boost * year_norm

    scores = scores_norm

    # Mask watched items
    if exclude_watched:
        watched = assets["user_watched"].get(user_idx, set())
        watched_t = torch.tensor(list(watched), dtype=torch.long, device=DEVICE)
        valid_mask = watched_t[watched_t < len(scores)]
        if len(valid_mask):
            scores[valid_mask] = float("-inf")

    # Genre filter — vectorised via pre-built genre string lookup
    if genre_filter:
        gf = genre_filter.lower()
        no_genre = torch.tensor(
            [i for i in range(len(scores))
             if gf not in assets["idx2genres_str"].get(i, "").lower()],
            dtype=torch.long, device=DEVICE
        )
        if len(no_genre):
            scores[no_genre] = float("-inf")

    # Top-K
    valid = (scores > float("-inf")).sum().item()
    k_actual = min(top_k, valid)
    if k_actual == 0:
        return []

    top_indices = torch.topk(scores, k=k_actual).indices.cpu().numpy()

    results = []
    for rank, idx in enumerate(top_indices, 1):
        idx  = int(idx)
        year = assets["idx2year"].get(idx, 0)
        results.append({
            "rank":   rank,
            "idx":    idx,
            "title":  assets["idx2title"].get(idx, f"Movie {idx}"),
            "score":  float(scores[idx]),
            "genres": assets["idx2genres_str"].get(idx, ""),
            "year":   int(year) if year else None,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_recommendations(recs: list[dict], user_idx: int,
                          mode: str, assets: dict,
                          show_history: bool = True,
                          recency_boost: float = 0.0):
    n_watched = len(assets["user_watched"].get(user_idx, set()))
    orig_id   = assets["idx2orig"].get(user_idx, user_idx)

    print(f"\n{'='*60}")
    print(f"  Recommendations for user {orig_id}  (internal idx: {user_idx})")
    boost_str = f"  |  recency_boost={recency_boost}" if recency_boost > 0 else ""
    print(f"  Mode: {mode.upper()}  |  Movies watched: {n_watched:,}{boost_str}")
    print(f"{'='*60}")

    if not recs:
        print("  No recommendations found (try --exclude_watched false or remove --genre filter)")
        return

    # Show user's top watched movies for context
    if show_history and n_watched > 0:
        train    = assets["train"]
        watched_df = train[train["user_idx"] == user_idx].nlargest(5, "rating")
        if len(watched_df):
            print("\n  Top rated by this user:")
            for _, row in watched_df.iterrows():
                title = assets["idx2title"].get(int(row["movie_idx"]), "?")
                print(f"    ★ {row['rating']:.1f}  {title}")
        print()

    print(f"  Top {len(recs)} recommendations:\n")
    for r in recs:
        genre_str = f" [{r["genres"]}]" if r["genres"] else ""
        print(f"  {r["rank"]:>3}. {r["title"]}{genre_str}")
        print(f"       score={r["score"]:.4f}")


def print_stats(assets: dict):
    s = assets["stats"]
    print(f"\n{'='*60}")
    print("  DataFlix Dataset Stats")
    print(f"{'='*60}")
    print(f"  Users:        {s['n_users']:>10,}")
    print(f"  Movies:       {s['n_movies']:>10,}")
    print(f"  Train ratings:{s['n_train']:>10,}")
    print(f"  Val ratings:  {s['n_val']:>10,}")
    print(f"  Test ratings: {s['n_test']:>10,}")
    print(f"  Cold-start:   {s.get('n_cold_start', '?'):>10}")
    print(f"  Density:      {s.get('density_pct', 0):>9.4f}%")
    print()

    # Model availability
    print("  Trained models:")
    for name, path in [("Path A (MSE)", MODEL_A_PATH), ("Path B (BPR)", MODEL_B_PATH)]:
        status = "✓ found" if path.exists() else "✗ not found"
        print(f"    {name}: {status}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="DataFlix recommendation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python recommend.py --user_id 42
  python recommend.py --user_id 42 --top_k 20 --mode b
  python recommend.py --user_id 42 --genre Action
  python recommend.py --user_id 42 --exclude_watched false
  python recommend.py --cold_start --top_k 10
  python recommend.py --stats
        """
    )
    p.add_argument("--user_id",          type=int,   default=None,
                   help="Original user ID (from Netflix/ML data)")
    p.add_argument("--user_idx",         type=int,   default=None,
                   help="Internal user index (alternative to --user_id)")
    p.add_argument("--top_k",            type=int,   default=10,
                   help="Number of recommendations (default: 10)")
    p.add_argument("--mode",             type=str,   default="ensemble",
                   choices=["a", "b", "ensemble"],
                   help="Scoring mode: a=MSE model, b=BPR model, ensemble=both (default)")
    p.add_argument("--genre",            type=str,   default=None,
                   help="Filter by genre keyword e.g. --genre Action")
    p.add_argument("--exclude_watched",  type=str,   default="true",
                   choices=["true", "false"],
                   help="Exclude already-watched movies (default: true)")
    p.add_argument("--cold_start",       action="store_true",
                   help="Recommend for a brand-new user with no history")
    p.add_argument("--stats",            action="store_true",
                   help="Print dataset and model stats then exit")
    p.add_argument("--no_history",       action="store_true",
                   help="Don't show user's watch history in output")
    p.add_argument("--recency_boost",    type=float, default=0.0,
                   help="Boost newer movies: 0.0=off, 0.3=moderate, 0.5=strong (default: 0.0)")
    return p.parse_args()


def resolve_user_idx(args, assets) -> int | None:
    """Resolve --user_id or --user_idx to internal index."""
    if args.user_idx is not None:
        if args.user_idx >= assets["n_users"]:
            print(f"[error] --user_idx {args.user_idx} out of range "
                  f"(max: {assets['n_users']-1})")
            return None
        return args.user_idx

    if args.user_id is not None:
        # Try both ML_ and NF_ prefixed IDs, and raw int
        candidates = [
            f"ML_{args.user_id}",
            f"NF_{args.user_id}",
            args.user_id,
            str(args.user_id),
        ]
        for c in candidates:
            if c in assets["orig2idx"]:
                return assets["orig2idx"][c]
        print(f"[error] user_id {args.user_id} not found.")
        print("  Tip: use --user_idx for the internal index directly.")
        print(f"  Valid range: 0 to {assets['n_users']-1}")
        return None

    return None


def main():
    args = parse_args()

    # Load assets (always needed)
    assets = load_assets()

    # Stats mode — no model needed
    if args.stats:
        print_stats(assets)
        return

    # Load models
    models = load_models(assets["n_users"], assets["n_movies"])

    # Cold start — use a synthetic user with zero history
    if args.cold_start:
        print("\n[Cold-start mode] Recommending based on content only (no user history)")
        # Use user_idx=0 as proxy but override history/features to zeros
        user_idx = 0
        assets_cs = dict(assets)
        assets_cs["history"]   = torch.zeros_like(assets["history"])
        assets_cs["user_feat"] = torch.zeros_like(assets["user_feat"])
        assets_cs["user_watched"] = {}  # no watched history

        import src.training.trainer as T
        T._HISTORY   = assets_cs["history"].to(DEVICE)
        T._USER_FEAT = assets_cs["user_feat"].to(DEVICE)

        recs = get_recommendations(
            user_idx, assets_cs, models,
            top_k=args.top_k, mode=args.mode,
            exclude_watched=False,
            genre_filter=args.genre,
            recency_boost=args.recency_boost
        )
        print_recommendations(recs, user_idx, args.mode, assets_cs,
                               show_history=False,
                               recency_boost=args.recency_boost)
        return

    # Normal mode — need a user
    if args.user_id is None and args.user_idx is None:
        print("[error] Provide --user_id, --user_idx, --cold_start, or --stats")
        print("  Run with --help for usage examples.")
        sys.exit(1)

    user_idx = resolve_user_idx(args, assets)
    if user_idx is None:
        sys.exit(1)

    exclude_watched = args.exclude_watched.lower() == "true"

    t0 = time.time()
    recs = get_recommendations(
        user_idx, assets, models,
        top_k=args.top_k,
        mode=args.mode,
        exclude_watched=exclude_watched,
        genre_filter=args.genre,
        recency_boost=args.recency_boost
    )
    elapsed = time.time() - t0

    print_recommendations(recs, user_idx, args.mode, assets,
                           show_history=not args.no_history,
                           recency_boost=args.recency_boost)
    print(f"  Inference time: {elapsed*1000:.1f}ms")


if __name__ == "__main__":
    main()